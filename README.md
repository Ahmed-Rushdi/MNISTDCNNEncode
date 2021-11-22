# MNISTDCNNEncode
training a Deep CNN model and using it to encode MNIST numbers 
### Encoding
28,28,1 to 7,7,4

### Decoding
7,7,4 to 28,28,1

## Model Structure

|Layer (type)                   |Output Shape       |Param #|
|-----------------------------------------------------------|
|input_1 (InputLayer)           |[(None, 28, 28, 1)]|0      |
|conv2d (Conv2D)                |(None, 28, 28, 100)|1000   |
|max_pooling2d (MaxPooling2D)   |(None, 14, 14, 100)|0      |
|conv2d_1 (Conv2D)              |(None, 14, 14, 20) |18020  |
|max_pooling2d_1 (MaxPooling2D) |(None, 7, 7, 20)   |0      |
|conv2d_2 (Conv2D)              |(None, 7, 7, 4)    |724    |
|up_sampling2d (UpSampling2D)   |(None, 14, 14, 4)  |0      |
|conv2d_3 (Conv2D)              |(None, 14, 14, 20) |740    |
|up_sampling2d_1 (UpSampling2D) |(None, 28, 28, 20) |0      |
|conv2d_4 (Conv2D)              |(None, 28, 28, 100)|18100  |
|conv2d_5 (Conv2D)              |(None, 28, 28, 1)  |401    |
=================================================================
Total params: 38,985
Trainable params: 38,985
Non-trainable params: 0
_________________________________________________________________
