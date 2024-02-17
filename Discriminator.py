import torch
from torch.utils.data import DataLoader,Dataset,random_split
import torch.nn as nn
from PIL import Image
import glob
from torchvision import transforms
import tqdm
import matplotlib.pyplot as plt


class Discriminator(nn.Module):
    def __init__(self,channel_size_input,channel_size_out):
        super(Discriminator,self).__init__()
        
        self.block1=nn.Sequential(nn.Conv2d(in_channels=channel_size_input+channel_size_out,out_channels=64,kernel_size=4,stride=2,padding=1,padding_mode="reflect"),
                                  nn.LeakyReLU(0.2))
        self.block2=self.Create_Block(in_channels=64,out_channels=128,stride=2)
        self.block3=self.Create_Block(in_channels=128,out_channels=256,stride=2)
        self.block4=self.Create_Block(in_channels=256,out_channels=512,stride=1)
        self.block5=self.Create_Block(in_channels=512,out_channels=512,stride=1)
        
        self.out_blockd=nn.Sequential(nn.Conv2d(in_channels=512,out_channels=1,kernel_size=1),
                                      nn.Sigmoid())
    
    def Create_Block(self,in_channels,out_channels,stride):
        block=nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=4,stride=stride,padding=1,padding_mode="reflect"),
                            nn.BatchNorm2d(out_channels),
                            nn.LeakyReLU(0.2))
        return block
        
    def forward(self,x,y):
        data=torch.concat([x,y],dim=1)
        x=self.block1(data)
        x=self.block2(x)
        x=self.block3(x)
        x=self.block4(x)
        x=self.block5(x)
        out_disc=self.out_blockd(x)
        
        return out_disc
        