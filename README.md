# PIX2PIS Model WITH UT-Zappos50K Dataset

## Introduction:
-In this project, I aimed to train a Pix2Pix model architecture for image to image translation.
-The aim of the education is to transform 1-dimensional gray shoe sketches into 3-dimensional RGB real shoe images.

## Tensorboard:
TensorBoard, along with saving training or prediction images, allows you to save them in TensorBoard and examine the changes graphically during the training phase by recording scalar values such as loss and accuracy. It's a very useful and practical tool.

## Dataset:
- I used the UT-Zappos50K dataset, and with the help of OpenCV, I obtained sketch images of the shoe data from this dataset and converted them to grayscale. 
- As a result, I have the UT-Zappos Dataset and an additional dataset containing drawings of each item in this dataset.
- I used the sketch data as input and the RGB real image data as output. 

- Dataset link: https://vision.cs.utexas.edu/projects/finegrained/utzap50k/

## Models:
- For training, I utilized the Pix2pix model, which I coded from scratch.
- This model is essentially like a conditional GAN, comprising a generator and a discriminator. However, the generator differs in that it's based on a U-net architecture. 
- The U-net operates similar to an encoder-decoder-based autoencoder.
- It takes an input, unsamples it to a certain feature map size, and then upsamples it back to the input size.
- An important feature of U-net is that, while passing from the encoder to the decoder, the same-sized connections from the encoder are sequentially concatenated to the input of the decoder
- The discriminator performs downsampling to image dimensions of (batch_size,1,30,30) for prediction.

- Pix2Pix Model link: https://arxiv.org/abs/1611.07004


## Train:
- In this model, 1-dimensional images with drawings are provided as input to the generator, and the generator is tasked with producing the real image of that drawing.
- Since the model is a type of Conditional GANs, both the input and output values are concatenated and given to the discriminator, and the output of discriminator is (1,30,30) shape.
- Before being fed into the model, the images are resized to (256x256) dimensions.
- We used Adam optimizer with learning_rate 5e-5 and betas (0.5,0.999)
- we used gradscaler to adjust gradients.
- We use BCElosswithLogits (because of without sigmoid in model) both generator and discriminator. In addition, we also use L1 _loss in the generator.


## Results:
- When I complete the update, there will be a graph of the images and values created on "Tensorboard" folder.

## Usage: 
- You can train the model by setting "TRAIN" to "True" in config file and your checkpoint will save in "config.CALLBACKS_PATH"
- Tensorboard files will created into "Tensorboard" folder during training time.
- Then you can generate the images from random noise by setting the "LOAD_CHECKPOINTS" and "TEST" values to "True" in the config file.

