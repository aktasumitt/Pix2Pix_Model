import torch,warnings,glob,dataset,config,train,testing,predict,Discriminator,Generator,checkpoints
from torch.utils.tensorboard import SummaryWriter
warnings.filterwarnings("ignore")

# Control GPU
devices=("cuda" if torch.cuda.is_available() else "cpu")

# Tensorboard
Tensorboard_Writer=SummaryWriter(config.TENSORBOARD_PATH)

# Create Custom Dataset
train_dataset=dataset.Datasets(input_paths=glob.glob(config.INPUT_PATHS+"/*"),
                        target_paths=glob.glob(config.TARGET_PATHS+"/*"))


# Ranadom Split
Train_dataset,Valid_dataset=dataset.Random_split(valid_size=config.VALID_SIZE,train_dataset=train_dataset)


# Create Dataloader
train_dataloader=dataset.Dataloader(dataset=Train_dataset,batch_size=config.BATCH_SIZE,shuffle=True)
valid_dataloader=dataset.Dataloader(dataset=Valid_dataset,batch_size=config.BATCH_SIZE,shuffle=False)


# Discriminator Model
Discriminator_Model=Discriminator.Discriminator(channel_size_input=config.CHANNEL_SIZE_INPUT,channel_size_out=config.CHANNEL_SIZE_OUTPUT)
Discriminator_Model.to(devices)

# Generator Model
Generator_Model=Generator.Generator(channel_size_input=config.CHANNEL_SIZE_INPUT,channel_size_out=config.CHANNEL_SIZE_OUTPUT)
Generator_Model.to(devices)


# Optimizers
optimizer_disc=torch.optim.Adam(Discriminator_Model.parameters(),lr=5e-4,betas=(0.5,0.999))
optimizer_gen=torch.optim.Adam(Generator_Model.parameters(),lr=config.LEARNING_RATE,betas=config.BETAS)

# Loss Functions
loss_BCE=torch.nn.BCEWithLogitsLoss()
loss_L1=torch.nn.L1Loss()

# Gradient Scalers
generator_scaler=torch.cuda.amp.GradScaler()
discriminator_scaler=torch.cuda.amp.GradScaler()


# Loading Checkpoints If You Have
if config.LOAD_CHECKPOINTS==True:
    Checkpoint=torch.load(f=config.SAVE_CHECKPOINT_PATH)
    resume_epoch=checkpoints.Load_Checkpoints(checkpoint=Checkpoint,
                                            model_gen=Generator_Model,model_disc=Discriminator_Model,
                                            optim_gen=optimizer_gen,optim_disc=optimizer_disc)

    print(f"Training will start from {resume_epoch}.epoch...\n")
else: 
    resume_epoch=0
    print(f"Training will start from scratch...\n")
    



# Train Models
if config.TRAIN==True:
    train.Training(EPOCHS=config.EPOCHS,dataloader=train_dataloader,Generator_Model=Generator_Model,
                    Discriminator_Model=Discriminator_Model,devices=devices,resume_epoch=resume_epoch,
                    discriminator_scaler=discriminator_scaler,generator_scaler=generator_scaler,
                    LAMDA=config.LAMDA_L1,loss_BCE=loss_BCE,loss_L1=loss_L1,max_batch=config.MAX_BATCH_TRAIN,
                    optimizer_disc=optimizer_disc,optimizer_gen=optimizer_gen,
                    Save_Checkpoints_fn=checkpoints.Save_Checkpoints,save_checkpoint_path=config.SAVE_CHECKPOINT_PATH,
                    tensorboard=Tensorboard_Writer)


# Test Models
if config.TEST==True:
    Loss_Test_Dict=testing.Test(test_dataloader=valid_dataloader,
                            generator_model=Generator_Model,
                            Discriminator_Model=Discriminator_Model,
                            lamda=config.LAMDA_L1,
                            l1_loss=loss_L1,
                            bce_loss=loss_BCE,
                            devices=devices,
                            max_batch=config.MAX_BATCH_TEST)


# Predict Your Custom Images From Edge to Realistic
if config.PREDICTIONS==True:
    predict.Predictions(generator_model=Generator_Model,
                        img_paths_list=glob.glob(config.PREDICT_PATHS+"/*"),
                        devices=devices)









