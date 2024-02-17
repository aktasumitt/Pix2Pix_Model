import torch
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
from PIL import Image
import glob
from torchvision import transforms
import tqdm
import matplotlib.pyplot as plt


class Generator(nn.Module):
    
    def __init__(self,channel_size_input,channel_size_out):
        super(Generator,self).__init__()
        
        self.channel_size_input=channel_size_input
        self.channel_size_out=channel_size_out
        
        self.initial_encoder_block=nn.Sequential(torch.nn.Conv2d(in_channels=self.channel_size_input,out_channels=64,kernel_size=4,stride=2,padding=1,padding_mode="reflect"),
                                           torch.nn.LeakyReLU(0.2))
        
        self.eblock2=self.Encoder_Block(in_channels=64,out_channels=128)
        self.eblock3=self.Encoder_Block(in_channels=128,out_channels=256)
        self.eblock4=self.Encoder_Block(in_channels=256,out_channels=512)
        self.eblock5=self.Encoder_Block(in_channels=512,out_channels=512)
        self.eblock6=self.Encoder_Block(in_channels=512,out_channels=512)
        self.eblock7=self.Encoder_Block(in_channels=512,out_channels=512)
        self.eblock8=self.Encoder_Block(in_channels=512,out_channels=512)
        
        self.dblock1=self.Decoder_Block(in_channels=512,out_channels=512)
        self.dblock2=self.Decoder_Block(in_channels=512*2,out_channels=1024)
        self.dblock3=self.Decoder_Block(in_channels=512*3,out_channels=1024)
        self.dblock4=self.Decoder_Block(in_channels=512*3,out_channels=1024)
        self.dblock5=self.Decoder_Block(in_channels=512*3,out_channels=1024)
        self.dblock6=self.Decoder_Block(in_channels=256*5,out_channels=512)
        self.dblock7=self.Decoder_Block(in_channels=128*5,out_channels=256)
        self.dblock8=self.Decoder_Block(in_channels=64*5,out_channels=128)
        
        self.out_dblock=nn.Sequential(nn.Conv2d(in_channels=128,out_channels=self.channel_size_out,kernel_size=1),
                                      nn.Tanh())

        
    def Encoder_Block(self,in_channels,out_channels):
        
        block=nn.Sequential(torch.nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=4,stride=2,padding=1,padding_mode="reflect"),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.LeakyReLU(0.2))
        
        return block
    
    
    def Decoder_Block(self,in_channels,out_channels):
        block=nn.Sequential(torch.nn.ConvTranspose2d(in_channels=in_channels,out_channels=out_channels,kernel_size=4,stride=2,padding=1),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU())
        return block


    def forward(self,data):
        # Encoder
        e1=self.initial_encoder_block(data)
        e2=self.eblock2(e1)
        e3=self.eblock3(e2)
        e4=self.eblock4(e3)
        e5=self.eblock5(e4)
        e6=self.eblock6(e5)
        e7=self.eblock7(e6)
        e8=self.eblock8(e7)

        # Decoder
        d1=self.dblock1(e8)
        d2=self.dblock2(torch.concat([d1,e7],dim=1))
        d3=self.dblock3(torch.concat([d2,e6],dim=1))
        d4=self.dblock4(torch.concat([d3,e5],dim=1))
        d5=self.dblock5(torch.concat([d4,e4],dim=1))
        d6=self.dblock6(torch.concat([d5,e3],dim=1))
        d7=self.dblock7(torch.concat([d6,e2],dim=1))
        d8=self.dblock8(torch.concat([d7,e1],dim=1))
        
        return self.out_dblock(d8)



