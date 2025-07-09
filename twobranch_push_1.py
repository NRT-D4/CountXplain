import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


from torchvision import transforms


import pytorch_lightning as pl


from dataset import CellDataset
from .counting_model import CSRNet


class countXplain(pl.LightningModule):
    def __init__(self, hparams, count_model):
        super().__init__()
        self.save_hyperparameters(hparams)


        # The counting model will be frozen
        for param in count_model.parameters():
            param.requires_grad = False


        self.front_end = count_model.frontend
        self.back_end = count_model.backend
        self.output_layer = count_model.output_layer


        self.prototypes = nn.Parameter(torch.rand(self.hparams["num_prototypes"], 64, 1, 1), requires_grad=True)


        # Required for l2 convolution
        self.ones = nn.Parameter(torch.ones(self.prototypes.shape),
                                 requires_grad=False)
       
        self.add_on_layers = nn.Sequential(
            # nn.Conv2d(in_channels=64, out_channels=self.prototypes.shape[1], kernel_size=1, stride=1),
            # nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.prototypes.shape[1], out_channels=self.prototypes.shape[1], kernel_size=1),
            nn.Sigmoid()
        ) # Check if 1 is enough




        self.counter = nn.Sequential(
            nn.Conv2d(self.prototypes.shape[0]//2, 1, kernel_size=1)
        )


#         # Initialize the counter with 1s
#         nn.init.constant_(self.counter[0].weight, 1e-6)


#         for param in self.counter.parameters():
#             param.requires_grad = False


        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.normal_(m.weight, std=0.01)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)




        
        self.fg_coef = self.hparams["fg_coef"]
        self.diversity_coef = self.hparams["diversity_coef"]
        self.proto_to_feature_coef =  self.hparams["proto_to_feature_coef"]
        
        

    def forward(self, x):
        # Get feaures from the front end
        x = self.front_end(x)
        x = self.back_end(x)
        x = self.add_on_layers(x)


        # Get the distance between the features and the prototypes
        distances = self._l2_convolution(x)


        # convert the distance to similarity
        similarity = self.distance2similarity(distances)




        # The first half of the similarity scores will be passed through the counter
        fg = self.counter(similarity[:, :self.prototypes.shape[0]//2, :, :])
        # fg = torch.mean(similarity[:, :self.prototypes.shape[0]//2, :, :], dim=1, keepdim=True)




       
        return x, fg, distances
   
    def diversity_loss(self):
        '''
        A method to calculate the diversity loss


        Returns:
            The diversity loss
        '''


        num_prototypes, _,_,_ = self.prototypes.shape


        prototypes = self.prototypes.view(num_prototypes, -1)
        norms = torch.norm(prototypes, dim=1,p=2, keepdim=True)
        prototypes = prototypes / norms


        dot_product = torch.mm(prototypes, prototypes.t())


        # mask out the diagonal because of self similarity
        mask = torch.ones_like(dot_product) - torch.eye(num_prototypes, device=dot_product.device)
        dot_product = dot_product * mask


        # Calculate the diversity loss
        diversity_loss = torch.sum(torch.abs(dot_product))


        return diversity_loss


    def distance2similarity(self, distance):
        '''
        A method to convert the distance to similarity


        Args:
            distance: The distance between the features and the prototypes


        Returns:
            The similarity between the features and the prototypes
        '''


        similarity = torch.log((distance + 1) / (distance + self.hparams["epsilon"]))
        return similarity
   
        # Method by InsightRNet
        # dist_max = self.prototypes.shape[1] * \
        #         self.prototypes.shape[2] * self.prototypes.shape[3]
        # return 1 / ((distance / dist_max) + self.hparams["epsilon"])


    def calculate_similarity(self, x):
        '''
        A method to measure the similarity with fixed inner product. (Adapted from https://github.com/cvlab-stonybrook/zero-shot-counting/blob/main/models/matcher.py)


        Args:
            x: The features from the front end


        Returns:
            The similarity between the features and the prototypes
        '''


        # Get the features shape
        bs, c, h, w = x.shape


        # Reshape the features to be of shape (bs, hw, c)
        features = x.flatten(2).permute(0, 2, 1)


        # Reshape the prototypes of shape (num_prototypes, c, hw)
        prototypes = self.prototypes.flatten(2)


        # Calculate the similarity between the features and the prototypes
        similarity = torch.bmm(features, prototypes)


    def _l2_convolution(self,x):
        '''
        A method to apply self.prototype vectors as L2 convolution filters on input x


        Args:
            x: The features from the front end


        Returns:
            The similarity between the features and the prototypes
        '''
        # print(f"Prototypes shape: {self.prototypes.shape}")
        # print(f"X shape: {x.shape}")
        # print(F"ones shape: {self.ones.shape}")




        # Get x^2
        x2 = x**2
        x2_patch_sum = F.conv2d(x2, weight=self.ones)


        protos = self.prototypes


        p2 = protos**2


        p2 = torch.sum(p2, dim=(1,2,3))


        p2_reshape = p2.view(-1, 1, 1)


        xp = F.conv2d(x, weight=protos)


        intermediate = -2*xp + p2_reshape


        distances = F.relu(x2_patch_sum + intermediate)


        return distances

    def data_coverage_loss(self,distances):
        '''
        A method to calculate the data coverage loss. It will find the minimum distance between the points in the feature map and the prototypes. This will ensure the points in the feature map are clustered around the prototypes.


        L = 1/N * sum(min(||x_n_i - p_m||^2))


        Args:
            distances: The distance from each prototype to each point in the feature map
        Returns:
            The data coverage loss
        '''
       
        # Find the minimum distance from each point in the feature map to each prototype
        min_distance, _ = torch.min(distances, dim=1)


        # Get the mean of the minimum distance
        mean_min_distance = torch.mean(min_distance)


        return mean_min_distance


    def orthonormality_loss_original(self):
        '''
        A method to calculate the orthonormality loss. It will calculate the dot product between the prototypes and ensure that they are orthogonal.


        L = sum(|p_m * p_n|)


        Returns:
            The orthonormality loss
        '''
        s_loss = 0


        for k in range(2):
            # Get the first half of the prototypes
            p_k = self.prototypes[k*self.prototypes.shape[0]//2:(k+1)*self.prototypes.shape[0]//2, :, :, :]
           
            # Get the mean of the prototypes
            p_k_mean = torch.mean(p_k, dim=0)


            # Normalize the prototypes
            p_k_2 = p_k - p_k_mean


            # Squeeze the prototypes
            p_k_2 = p_k_2.squeeze().squeeze()


            # Calculate the dot product between the prototypes
            p_k_dot = p_k_2 @ p_k_2.T


            # Mask the diagonal
            s_matrix = p_k_dot - (torch.eye(p_k.shape[0], device=p_k_dot.device))


            # Calculate the orthonormality loss
            s_loss += torch.norm(s_matrix, p=2)


        return s_loss/2
    
    def orthonormality_loss(self, similarity_threshold=0.8):
        '''
        Calculate a soft diversity loss that allows prototypes to be similar
        but penalizes extreme similarity.
        '''
        num_prototypes = self.prototypes.shape[0]
        assert num_prototypes % 2 == 0, "Number of prototypes must be even"
        half_count = num_prototypes // 2

        total_loss = 0

        for k in range(2):
            # Get half of the prototypes
            p_k = self.prototypes[k*half_count:(k+1)*half_count, :, 0, 0]
            
            # Normalize the prototypes
            p_k_normalized = F.normalize(p_k, p=2, dim=1)
            
            # Calculate the cosine similarity between the prototypes
            similarity_matrix = torch.mm(p_k_normalized, p_k_normalized.t())
            
            # Create a mask to ignore self-similarity
            mask = torch.eye(half_count, device=similarity_matrix.device)
            
            # Compute loss only for similarities above the threshold
            excess_similarity = F.relu(similarity_matrix - similarity_threshold)
            
            # Sum the excess similarities, ignoring self-similarity
            half_loss = torch.sum(excess_similarity * (1 - mask))
            
            # Normalize by the number of off-diagonal elements
            half_loss = half_loss / (half_count * (half_count - 1))
            
            total_loss += half_loss

        return total_loss / 2


    def inter_class_loss(self):

        # prototypes need to be reshaped to (num_prototypes, d*h*w)
        flattened_prototypes = self.prototypes.view(self.prototypes.shape[0], -1)


        # Getting the foreground and background prototypes
        fg_prototypes = flattened_prototypes[:self.prototypes.shape[0] // 2, :]
        bg_prototypes = flattened_prototypes[self.prototypes.shape[0] // 2:, :]


        fg_centroid = torch.mean(fg_prototypes, dim=0)
        bg_centroid = torch.mean(bg_prototypes, dim=0)


        # Calculate the inter class loss
        inter_class_loss = torch.norm(fg_centroid - bg_centroid, p=2)


        return inter_class_loss


    def proto_to_feature_loss(self,fmaps,distances, label):
        '''
        A modified version of the proto_to_feature loss.

        '''
        bg_loss, cell_loss = 0, 0
        batch_size, num_prototypes, h, w = distances.shape

        single_channel_fmaps = label

        flattened_fmaps = single_channel_fmaps.view(batch_size, -1) 
        flattened_distances = distances.view(batch_size, num_prototypes, -1)

        for i in range(num_prototypes):
            if i < num_prototypes//2:

                max_indices = torch.argmax(flattened_fmaps, dim=1)

                dist = flattened_distances[range(batch_size), i, max_indices]

                cell_loss += torch.mean(dist)

            else:
                min_indices = torch.argmin(flattened_fmaps, dim=1)

                dist = flattened_distances[range(batch_size), i, min_indices]

                bg_loss += torch.mean(dist)

        return bg_loss, cell_loss


    def training_step(self, batch, batch_idx):
        x, y = batch

        # FWD pass
        fmaps, fg, distances = self(x)


        # calculate the losses
        # Counting loss
        fg_loss = F.mse_loss(fg*100, y*100, reduction='mean')


        counting_loss = F.l1_loss(fg.sum(dim = (2,3)), y.sum(dim = (2,3)), reduction='mean')


        # Diversity loss
        diversity_loss = self.orthonormality_loss()
       
        # Prototype to feature loss
        bg_loss, cell_loss = self.proto_to_feature_loss(fmaps,distances, y)


        # Total loss
        loss =  self.fg_coef * fg_loss + self.diversity_coef * diversity_loss + self.proto_to_feature_coef * \
            (bg_loss + cell_loss)


        # Show a warning if any of the losses are NaN
        if torch.isnan(loss):
            print(f"Loss is NaN")
            print(f"Number of NaN values in dmap: {torch.sum(torch.isnan(bg))}")
            print(f"Number of NaN values in dmap: {torch.sum(torch.isnan(fg))}")
            print(f"Number of NaN values in y: {torch.sum(torch.isnan(y))}")
   


        self.log('train_loss', loss)
        self.log('train_counting_loss', counting_loss)
        self.log('train_fg_loss', fg_loss)
        self.log('train_diversity_loss', diversity_loss)
        self.log('train_bg_loss', bg_loss)
        self.log('train_cell_loss', cell_loss)


        return loss
   
    def validation_step(self, batch, batch_idx):
        x, y = batch

        # Fwd pass
        fmaps,fg, distances = self(x)


        # calculate the losses
        # Counting loss


        fg_loss = F.mse_loss(fg*100, y*100, reduction='mean')


        counting_loss = F.l1_loss(fg.sum(dim = (2,3)), y.sum(dim = (2,3)), reduction='mean')
   
        # Diversity loss
        diversity_loss = self.orthonormality_loss()


        # Prototype to feature loss
        bg_loss, cell_loss = self.proto_to_feature_loss(fmaps,distances, y)

        # loss = self.delta * bg_loss + self.alpha * fg_loss + self.beta * diversity_loss + self.gamma * proto_to_feature_loss + self.delta * data_coverage_loss + self.beta * inter_class

        loss = self.fg_coef * fg_loss + self.diversity_coef * diversity_loss + self.proto_to_feature_coef * \
            (bg_loss + cell_loss)


        self.log('val_loss', loss)
        self.log('val_counting_loss', counting_loss)
        self.log('val_fg_loss', fg_loss)
        self.log('val_diversity_loss', diversity_loss)
        self.log('val_bg_loss', bg_loss)
        self.log('val_cell_loss', cell_loss)


        return loss
   
    def test_step(self, batch, batch_idx):
        x, y = batch

        # Fwd pass
        fmaps,fg, distances = self(x)

        # calculate the losses
        # Counting loss
        fg_loss = F.mse_loss(fg, y, reduction='mean')

        counting_loss = F.l1_loss(fg.sum(dim = (2,3)), y.sum(dim = (2,3)), reduction='mean')

        # Diversity loss
        diversity_loss = self.orthonormality_loss()

        # # Prototype to feature loss
        bg_loss, cell_loss = self.proto_to_feature_loss(fmaps,distances, y)

        # loss = self.delta * bg_loss + self.alpha * fg_loss + self.beta * diversity_loss + self.gamma * proto_to_feature_loss + self.delta * data_coverage_loss + self.beta * inter_class
        loss = self.fg_coef * fg_loss + self.diversity_coef * diversity_loss + self.proto_to_feature_coef * \
            (bg_loss + cell_loss)


        self.log('test_loss', loss)
        self.log('test_fg_loss', fg_loss)
        self.log('test_counting_loss', counting_loss)
        self.log('test_diversity_loss', diversity_loss)
        self.log('test_bg_loss', bg_loss)
        self.log('test_cell_loss', cell_loss)


        return loss
   
    def on_validation_epoch_end(self):
        if self.hparams["dataset"] == "DCC":
            push_epoch = 100
        elif self.hparams["dataset"] == "IDCIA_v1":
            push_epoch = 50
    
        if self.current_epoch % push_epoch == 0 and self.current_epoch != 0:
            pushproto = PushPrototypes(self)
            pushproto.push_prototypes(self.train_dataloader())

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams["lr"],weight_decay=5*1e-4)
        
        optimizer = torch.optim.Adam([
            {'params': self.front_end.parameters(), 'lr': 1e-7},
            {'params': self.back_end.parameters(), 'lr': 1e-7},
            {'params': self.add_on_layers.parameters(), 'lr': self.hparams["lr"]},
            {'params': self.prototypes, 'lr': self.hparams["lr"]},
            {'params': self.counter.parameters(), 'lr': self.hparams["lr"]}
        ], lr=self.hparams["lr"])
        
        # optimizer = torch.optim.AdamW([
        #     {'params': self.front_end.parameters(), 'lr': 1e-7,},
        #     {'params': self.back_end.parameters(), 'lr': 1e-7},
        #     {'params': self.add_on_layers.parameters(), 'lr': 1e-2},
        #     {'params': self.prototypes, 'lr': 1e-2},
        #     {'params': self.counter.parameters(), 'lr': 1e-2}
        # ], lr=self.hparams["lr"], weight_decay=5*1e-4)
        
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams["lr"], momentum=1e-4, weight_decay=5*1e-4)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=0.1, cycle_momentum=False)
        return [optimizer]
   
    def prepare_data(self):
        
        self.train_dataset = CellDataset(self.hparams["train_dir"], transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]))
        self.val_dataset = CellDataset(self.hparams["val_dir"], transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]))
        self.test_dataset = CellDataset(self.hparams["test_dir"], transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]))
        
        # Merge the train, val and test datasets and split them into train, val and test
        self.dataset = torch.utils.data.ConcatDataset([self.train_dataset, self.val_dataset, self.test_dataset])

        train_size = 100
        val_size = 76
        

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.dataset, [train_size, val_size])
        self.test_dataset = self.val_dataset


    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.hparams["batch_size"], shuffle=True, num_workers=self.hparams["num_workers"])
   
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.hparams["batch_size"], shuffle=False, num_workers=self.hparams["num_workers"])
   
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.hparams["batch_size"], shuffle=False, num_workers=self.hparams["num_workers"])
   
class PushPrototypes:
    def __init__(self, model):

        self.model = model
        self.model.eval()

        self.num_prototypes = self.model.prototypes.shape[0]
        self.proto_shape  = self.model.prototypes.shape

    def push_prototypes(self,dataloader):
        self.init_save_variables()

        for (batch_im,_) in dataloader:
            self.update_protos_batch(batch_im.to(self.model.device))

        self.update_model_variables()

    def init_save_variables(self):
        self.global_min_dist = np.full(self.num_prototypes, np.inf)
        
        self.proto_latent_repr = np.zeros((self.num_prototypes, 64,1,1))

        self.proto_images = np.zeros((self.num_prototypes, 3, 256, 256), dtype=np.uint8)

    def update_protos_batch(self, batch):
        with torch.no_grad():
            batch_fmaps, batch_fg, batch_distances = self.model(batch)

        batch_fmaps = batch_fmaps.detach().cpu().numpy()
        batch_distances = batch_distances.detach().cpu().numpy()

        # get all minimum distances to prototypes and corresponding indices
        proto_first = batch_distances.swapaxes(0, 1)
        dist_perproto = proto_first.reshape(self.num_prototypes, -1)

        min_dist = np.min(dist_perproto, axis=1)
        min_idx = np.argmin(dist_perproto, axis=1)
        index_unravel = np.unravel_index(min_idx, proto_first.shape[1:])

        whole_batch_im = batch[index_unravel[0]]

        for proto_j in range(self.num_prototypes):
            if self.global_min_dist[proto_j] > (newdist := min_dist[proto_j]):
                batch_index = (index_unravel[0][proto_j], index_unravel[1][proto_j], index_unravel[2][proto_j])

                self.global_min_dist[proto_j] = newdist

                if batch_fmaps.shape[2:] == batch_distances.shape[2:]:
                    latent_repr = batch_fmaps[batch_index[0], :, batch_index[1], batch_index[2]]

                    self.proto_latent_repr[proto_j] = latent_repr.reshape(-1,1,1)
                else:
                    latent_repr = batch_fmaps[batch_index[0], :,:,:]
                    self.proto_latent_repr[proto_j] = latent_repr

               

    def update_model_variables(self):

        self.model.prototypes.data.copy_(torch.tensor(self.proto_latent_repr, dtype=torch.float32))
        self.model.prototypes.requires_grad = True






