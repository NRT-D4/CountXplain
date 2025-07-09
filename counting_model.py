from typing import Any, List, Optional, Union
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Tuple


class CSRNet(pl.LightningModule):

    def __init__(self, learning_rate=1e-6):
        super().__init__()

        # Define layer configuration
        # M stands for MaxPooling2D
        self.frontend_feats = [
            64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feats = [512, 512, 512, 256, 128, 64]

        # Define block of layers
        self.frontend = self.make_layers(self.frontend_feats)
        self.backend = self.make_layers(
            self.backend_feats, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        self.learning_rate = learning_rate

        # Load weights
        vgg16 = models.vgg16(weights = 'DEFAULT')
        self._initialize_weights()

        # Fetch part of pretrained model
        vgg16_dict = vgg16.state_dict()
        frontend_dict = self.frontend.state_dict()
        transfer_dict = {k: vgg16_dict['features.' + k] for k in frontend_dict}

        # Transfer weights
        self.frontend.load_state_dict(transfer_dict)

        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def make_layers(config, in_channels=3, batch_norm=False, dilation=False):
        d_rate = 2 if dilation else 1
        layers = []

        for filters in config:
            if filters == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                continue

            conv2d = nn.Conv2d(in_channels, filters,
                            kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(filters), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]

            in_channels = filters
        return nn.Sequential(*layers)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        print(f"y_hat: {y_hat.shape}")
        print(f"y: {y.shape}")

        loss = self.loss(y_hat, y.unsqueeze(1))
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        abs_diff = torch.abs(y_hat.sum(dim=(2,3)) - y.sum(dim=(2,3)))
        return {'abs_diff': abs_diff}
    
    def validation_epoch_end(self, outputs):
        mae = torch.cat([x['abs_diff'] for x in outputs]).mean()
        self.log('val_mae', mae, prog_bar=True)
        return mae

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        abs_diff = torch.abs(y_hat.sum(dim=(2,3)) - y.sum(dim=(2,3)))
        return {'abs_diff': abs_diff}
    
    def test_epoch_end(self, outputs):
        mae = torch.cat([x['abs_diff'] for x in outputs]).mean()
        self.log('test_mae', mae)
        return mae
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        return optimizer

        


def conv_block(channels: Tuple[int, int],
               size: Tuple[int, int],
               stride: Tuple[int, int]=(1, 1),
               N: int=1):
    """
    Create a block with N convolutional layers with ReLU activation function.
    The first layer is IN x OUT, and all others - OUT x OUT.

    Args:
        channels: (IN, OUT) - no. of input and output channels
        size: kernel size (fixed for all convolution in a block)
        stride: stride (fixed for all convolution in a block)
        N: no. of convolutional layers

    Returns:
        A sequential container of N convolutional layers.
    """
    # a single convolution + batch normalization + ReLU block
    block = lambda in_channels: nn.Sequential(
        nn.Conv2d(in_channels=in_channels,
                  out_channels=channels[1],
                  kernel_size=size,
                  stride=stride,
                  bias=False,
                  padding=(size[0] // 2, size[1] // 2)),
        nn.BatchNorm2d(num_features=channels[1]),
        nn.ReLU()
    )
    # create and return a sequential container of convolutional layers
    # input size = channels[0] for first block and channels[1] for all others
    return nn.Sequential(*[block(channels[bool(i)]) for i in range(N)])


class FCRN_A(pl.LightningModule):
    def __init__(self, config):
        super(FCRN_A, self).__init__()

        self.hyperparameters = config

        
        self.feature_extractor = nn.Sequential( 
            # downsampling
            conv_block(channels=(3, 32), size=(3, 3), N=2),
            nn.MaxPool2d(2),

            conv_block(channels=(32, 64), size=(3, 3), N=2),
            nn.MaxPool2d(2),

            conv_block(channels=(64, 128), size=(3, 3), N=2),
            nn.MaxPool2d(2),

            # "convolutional fully connected"
            conv_block(channels=(128, 512), size=(3, 3), N=2)
        )

        self.regressor = nn.Sequential(
            # upsampling
            nn.Upsample(scale_factor=2),
            conv_block(channels=(512, 128), size=(3, 3), N=2),

            nn.Upsample(scale_factor=2),
            conv_block(channels=(128, 64), size=(3, 3), N=2),

            nn.Upsample(scale_factor=2),
            conv_block(channels=(64, 1), size=(3, 3), N=2),
        )

        self.model = nn.Sequential(
            self.feature_extractor,
            self.regressor
        )

        # self.init_weights()

        self.loss = nn.MSELoss(reduction='mean')


        self.val_diffs = []
        self.test_diffs = []

    def init_weights(self):
        # initialize weights of the model with an orthogonal initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)  
        self.log('train_loss', loss)
        return loss
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        abs_diff = torch.abs((y_hat.sum(dim=(2, 3))/100) - (y.sum(dim=(2, 3))/100))
        loss = self.loss(y_hat, y)
        self.val_diffs.append(abs_diff)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        abs_diff = torch.abs((y_hat.sum(dim=(2, 3))/100) - (y.sum(dim=(2, 3))/100))

        loss = self.loss(y_hat, y)
        self.test_diffs.append(abs_diff)
        return loss
    
    def on_validation_epoch_end(self):
        mae = torch.cat([x for x in self.val_diffs]).mean()
        self.log('val_mae', mae, prog_bar=True)
        self.val_diffs = []
        return mae
        
    def on_test_epoch_end(self):
        mae = torch.cat([x for x in self.test_diffs]).mean()
        self.log('test_mae', mae, prog_bar=True)
        self.test_diffs = []
        return mae


    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hyperparameters['learning_rate'], momentum=0.9, weight_decay=5e-4)
        # step_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        return [optimizer]
