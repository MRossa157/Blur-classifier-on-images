# Image blur classifier
This CNN is trained to to find the blur on the image. When receiving a photo as input, the model will return the percentage of blurring of the photo

## Model Architecture
This CNN model is built using the PyTorch library's. The architecture is divided into two parts: the convolutional neural network (cnn) and the fully connected (fc) layers.

The cnn section consists of 5 convolutional layers and max pooling layers. The input image has a size of [3, 640, 640] and goes through the following sequence of operations: 
- The first convolutional layer has 64 filters with a kernel size of 3, stride of 1 and padding of 1, resulting in an output size of [64, 640, 640]
- Batch normalization is applied to the output of the first convolutional layer
- A ReLU activation function is applied
- The output is then passed through a max pooling layer with a kernel size of 2, stride of 2, and padding of 0, resulting in an output size of [64, 320, 320]
- This sequence of convolution, batch normalization, ReLU, and max pooling is repeated for 4 more times with increasing number of filters and reducing spatial resolution.

The fc section takes the output of cnn section and applies a sequence of linear and ReLU layers to produce the final output.

## Training
The model is trained using Cross Entropy Loss and Adam optimizer with a learning rate of 0.001. The model was trained on a dataset of images of blur and non-blur photo.

## Evaluation
The performance of this model is evaluated by AUC-ROC metric which is 92%.

## Usage
To use the model, input a .jpg image into the model and the model will will return the percentage of blurring of the photo. 

You can use **prod.py** to use this model with a user-friendly interface.

## Model download
The trained model can be downloaded from the link below:

https://drive.google.com/file/d/1o0R_RUrBrHFDlD2nk2pclRDtrjq9muAa/view?usp=share_link
