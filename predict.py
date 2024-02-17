import torch
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
from PIL import Image
import glob
from torchvision import transforms
import tqdm
import matplotlib.pyplot as plt


def Predictions(generator_model,img_paths_list,devices):
    image_list=[]
    
    for path in range(len(img_paths_list)):
        
        img=Image.open(path)
        img=transforms.Compose([transforms.ToTensor(),
                            transforms.Resize((256,256)),
                            transforms.Normalize(((0.5,),(0.5,)))])(img)

        image_list.append(img)
    
    image_list=torch.stack(image_list).to(device=devices)
    
    img_predict=generator_model(image_list)
    img_predict=img_predict.cpu().detach()
    
    for i in range(len(img_predict)):
        plt.subplot(int(len(img_predict)/2),2,i+1)
        plt.imshow(torch.transpose(torch.transpose(img_predict[i],0,2),0,1))
    
    plt.show()
    
    

