import torch
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
from PIL import Image
import glob
from torchvision import transforms
import tqdm
import matplotlib.pyplot as plt



def Test(test_dataloader,generator_model,bce_loss,l1_loss,Discriminator_Model,lamda,max_batch,devices):
        disc_total_loss=0.0
        gen_total_loss=0.0
        
        with torch.no_grad(): 
            for batch,(x,y) in enumerate(test_dataloader):
                
                x=x.to(devices)
                y=y.to(devices)
                
                y_fake=generator_model(x)
                d_fake=Discriminator_Model(x,y_fake.detach())
                d_real=Discriminator_Model(x,y)
                
                l1_gloss=l1_loss(y_fake,y)*lamda
                loss_fake=bce_loss(d_fake,torch.zeros_like(d_fake))
                loss_real=bce_loss(d_real,torch.ones_like(d_fake))
                
                loss_discriminator=(loss_real+loss_fake)/2
                loss_generator=loss_fake+l1_gloss
                
                disc_total_loss+=loss_discriminator
                gen_total_loss+=loss_generator
                
                if batch==max_batch:
                    break
                
            disc_loss={"Generator_test_loss":gen_total_loss/(batch+1),
                        "Discriminator_test_loss":disc_total_loss/(batch+1)}
            
        print(disc_loss)
            
        y_fake=y_fake.cpu().detach()
        y=y.cpu().detach()
            
        for i in range(len(y_fake)):
            plt.subplot(int(len(y_fake)/2),2,i+1)
            plt.imshow(torch.transpose(torch.transpose(y_fake[i],0,2),0,1))
            plt.title("Predicted_image")
        plt.show()
            
        for i in range(len(y)):
            plt.subplot(int(len(y)/2),2,i+1)
            plt.imshow(torch.transpose(torch.transpose(y[i],0,2),0,1))
            plt.title("Real_İmage")
            
        plt.show()
        
        
        return disc_loss
    
