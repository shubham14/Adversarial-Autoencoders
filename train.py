import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from data_loader import *
from config import *
from model import *
import os

data_load = DataLoader(64)
dl = data_load.loadTrainData()

def reset_grad(P, Q, D):
    Q.zero_grad()
    P.zero_grad()
    D.zero_grad()
    
def train(P, Q, D, batch_size, nz, lr=0.001, use_cuda=False):
    Q_solver = optim.Adam(Q.parameters(), lr=lr)
    P_solver = optim.Adam(P.parameters(), lr=lr)
    D_solver = optim.Adam(D.parameters(), lr=lr*0.1)
    for it in range(100000):
        for batch_idx, batch_item in enumerate(dl):
            #X = sample_X(mb_size)
            """ Reconstruction phase """
            X = Variable(batch_item[0])
            if use_cuda:
                X = X.cuda()
    
            z_sample = Q(X)
    
            X_sample = P(z_sample)
            recon_loss = F.mse_loss(X_sample, X)
    
            recon_loss.backward()
            P_solver.step()
            Q_solver.step()
            reset_grad(P, Q, D)
    
            """ Regularization phase """
            # Discriminator
            for _ in range(5):
                z_real = Variable(torch.randn(batch_size, nz))
                if use_cuda:
                    z_real = z_real.cuda()    
                z_fake = Q(X).view(batch_size,-1)    
                D_real = D(z_real)
                D_fake = D(z_fake)    
                D_loss = -torch.mean(torch.log(D_real) + torch.log(1 - D_fake))    
                D_loss.backward()
                D_solver.step()    
                reset_grad(P, Q, D)
    
            # Generator
            for _ in range(5):
                z_fake = Q(X).view(batch_size,-1)
                D_fake = D(z_fake)        
                G_loss = -torch.mean(torch.log(D_fake))        
                G_loss.backward()
                Q_solver.step()
                reset_grad(P, Q, D)
    
            if batch_idx % 10 == 0:
                print('Iter-{}; D_loss: {:.4}; G_loss: {:.4}; recon_loss: {:.4}'
                      .format(batch_idx, D_loss.data[0], G_loss.data[0], recon_loss.data[0]))
                torch.save(Q,"Q_latest.pth")
                torch.save(P,"P_latest.pth")
                torch.save(D,"D_latest.pth")
    
if __name__ == "__main__":
    data_load = DataLoader(64)
    tr_loader = data_load.loadTrainData()
    l = []
    for x in tr_loader:
        l.append(x[0])
    
    inp_shape = l[0].shape
    latent_size = 100
    hidden_dim = 200
    class_size=10
    enc_cfg = Enc_cfg()
    dec_cfg = Dec_cfg()
    Q = Encoder(inp_shape, latent_size, class_size, enc_cfg)
    P = Decoder(inp_shape, latent_size, class_size, dec_cfg)
    D = Discriminator(latent_size, hidden_dim)
    train(P, Q, D, 64, 100)