import torch
from torch.utils.data import DataLoader,Dataset,random_split
import torch.nn as nn
from PIL import Image
import glob
from torchvision import transforms
import tqdm
import matplotlib.pyplot as plt



class Datasets(Dataset):
    def __init__(self,input_paths,target_paths):
        
        self.input_paths=input_paths
        self.target_paths=target_paths
        self.transformer=transforms.Compose([transforms.ToTensor(),
                                             transforms.Resize((256,256)),
                                             transforms.Normalize((0.5,),(0.5,))])
    
    def __len__(self):
        return len(self.input_paths)
        
    def __getitem__(self,index):
        

        edges_img=Image.open(self.input_paths[index])
        edges_img=self.transformer(edges_img)
        
        
        real_img=Image.open(self.target_paths[index])
        real_img=self.transformer(real_img)
            
            
        return edges_img, real_img
            
            
        
def Random_split(valid_size,train_dataset):
    valid_dataset,train_dataset=random_split(dataset=train_dataset,lengths=[valid_size,len(train_dataset)-valid_size])
    
    return train_dataset,valid_dataset

def Dataloader(dataset,batch_size,shuffle:bool):
    dataloader=DataLoader(dataset=dataset,batch_size=batch_size,shuffle=shuffle)
    return dataloader
    