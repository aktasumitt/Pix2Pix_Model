import torch
from torch.utils.data import DataLoader,Dataset
import tqdm
from torchvision.utils import make_grid

def Training(EPOCHS,resume_epoch,dataloader,Generator_Model,Discriminator_Model,devices,optimizer_disc,optimizer_gen,loss_L1,loss_BCE,discriminator_scaler,generator_scaler,LAMDA,max_batch,Save_Checkpoints_fn,tensorboard,save_checkpoint_path):
    
    for epoch in range(resume_epoch,EPOCHS):
        
        Progress_Bar=tqdm.tqdm(range(max_batch),"Training Progress",leave=True)
        loss_total_generator=0.0
        loss_total_discriminator=0.0
        
        for batch,(x,y) in enumerate(dataloader):
            
            x=x.to(devices)
            y=y.to(devices)
            
            with torch.cuda.amp.autocast():
                y_fake=Generator_Model(x)
                D_real=Discriminator_Model(x,y)
                D_fake=Discriminator_Model(x,y_fake.detach())
                
                loss_real=loss_BCE(D_real,torch.ones_like(D_real))
                loss_fake=loss_BCE(D_fake,torch.zeros_like(D_fake))
                loss_discriminator=(loss_real+loss_fake)/2
                
                
                Discriminator_Model.zero_grad()
                discriminator_scaler.scale(loss_discriminator).backward()
                discriminator_scaler.step(optimizer_disc)
                discriminator_scaler.update()
                
            with torch.cuda.amp.autocast():
                
                g_fake=Discriminator_Model(x,y_fake)
                fake_gloss=loss_BCE(g_fake,torch.ones_like(g_fake))
                l1_gloss=loss_L1(y_fake,y)*LAMDA
                
                loss_Generator=fake_gloss+l1_gloss
                
                optimizer_gen.zero_grad()
                generator_scaler.scale(loss_Generator).backward()
                generator_scaler.step(optimizer_gen)
                generator_scaler.update()
            
            loss_total_discriminator+=loss_discriminator.item()
            loss_total_generator+=loss_Generator.item()
            
            if batch%10==0:
                
                Progress_Bar.set_postfix({"Epoch": epoch+1,
                                         "Loss Generator": loss_total_generator/(batch+1),
                                        "Loss Discriminator": loss_total_discriminator/(batch+1)})
            Progress_Bar.update(1)
            
            if batch==max_batch:
               break             

        
        image_edge=make_grid(x,nrow=10)
        image_real=make_grid(y,nrow=10)
        image_generate=make_grid(y_fake,nrow=10)
        
        tensorboard.add_image("Edge images",image_edge,global_step=epoch+1)
        tensorboard.add_image("Real images",image_real,global_step=epoch+1)
        tensorboard.add_image("Generated images",image_generate,global_step=epoch+1)
        
        tensorboard.add_scalar("Loss_Generator",(loss_total_generator/(batch+1)),global_step=epoch+1)
        tensorboard.add_scalar("Loss_Discriminator",(loss_total_generator/(batch+1)),global_step=epoch+1)
        
        Save_Checkpoints_fn(optim_gen=optimizer_gen,
                            optim_disc=optimizer_disc,
                            model_gen=Generator_Model,
                            model_disc=Discriminator_Model,
                            epoch=epoch+1,
                            save_path=save_checkpoint_path)
        

        
    
