import torch
import torch.nn as nn
import torch.nn.functional as F

class Tripletnet(nn.Module):
    def __init__(self, embeddingnet):
        super(Tripletnet, self).__init__()
        self.embeddingnet = embeddingnet

    def forward(self, x, y, z):
        recon_batch_x, embedded_x = self.embeddingnet(x)
        recon_batch_y, embedded_y = self.embeddingnet(y)
        recon_batch_z, embedded_z = self.embeddingnet(z)
        dist_a = F.pairwise_distance(embedded_x, embedded_y, 2)
        dist_b = F.pairwise_distance(embedded_x, embedded_z, 2)
        return dist_a, dist_b,recon_batch_x,recon_batch_y,recon_batch_z ,embedded_x, embedded_y, embedded_z
