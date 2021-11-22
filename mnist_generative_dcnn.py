# %% [markdown]
# # 20180008 - 20180256
# # Autoencoder

# %% [code] {"execution":{"iopub.status.busy":"2021-11-21T20:36:27.829579Z","iopub.execute_input":"2021-11-21T20:36:27.83018Z","iopub.status.idle":"2021-11-21T20:36:34.994945Z","shell.execute_reply.started":"2021-11-21T20:36:27.830086Z","shell.execute_reply":"2021-11-21T20:36:34.994213Z"}}
import keras
from keras import layers

embed_filter_c = 4

input_img = keras.Input(shape=(28, 28, 1))

x = layers.Conv2D(100, (3, 3), activation=layers.LeakyReLU(), padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x) #(28,28 to 14,14)

x = layers.Conv2D(20, (3, 3), activation=layers.LeakyReLU(), padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)

encoded = layers.Conv2D(embed_filter_c, (3, 3), activation=layers.LeakyReLU(),padding='same')(x)

x = layers.UpSampling2D((2, 2))(encoded)
x = layers.Conv2D(20, (3, 3), activation=layers.LeakyReLU(),padding='same')(x)

x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(100, (3, 3), activation=layers.LeakyReLU(),padding='same')(x)

decoded = layers.Conv2D(1, (2, 2), activation=layers.LeakyReLU(), padding='same')(x)
print("Model Structure\n------------------------------------")
autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

# %% [code] {"execution":{"iopub.status.busy":"2021-11-21T20:36:34.997555Z","iopub.execute_input":"2021-11-21T20:36:34.997797Z","iopub.status.idle":"2021-11-21T20:39:27.489989Z","shell.execute_reply.started":"2021-11-21T20:36:34.997768Z","shell.execute_reply":"2021-11-21T20:39:27.489274Z"}}
from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

autoencoder.fit(x_train, x_train,
                epochs=30, #5, #50,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test))
#                 ,verbose =0)
                #callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

# %% [markdown]
# # Encoder

# %% [code] {"execution":{"iopub.status.busy":"2021-11-21T20:39:27.49119Z","iopub.execute_input":"2021-11-21T20:39:27.491529Z","iopub.status.idle":"2021-11-21T20:39:28.264806Z","shell.execute_reply.started":"2021-11-21T20:39:27.491492Z","shell.execute_reply":"2021-11-21T20:39:28.264037Z"}}
encoder = keras.Model(input_img, encoded)

encoded_imgs = encoder.predict(x_test)

print("Encoder Structure\n------------------------------------")
encoder.summary()

# %% [markdown]
# # Decoder

# %% [code] {"execution":{"iopub.status.busy":"2021-11-21T20:39:28.266894Z","iopub.execute_input":"2021-11-21T20:39:28.267406Z","iopub.status.idle":"2021-11-21T20:39:28.753436Z","shell.execute_reply.started":"2021-11-21T20:39:28.267349Z","shell.execute_reply":"2021-11-21T20:39:28.752666Z"}}
decoder_input = keras.Input(shape=(7,7,embed_filter_c))
n5 = autoencoder.layers[-5](decoder_input)
n4 = autoencoder.layers[-4](n5)
n3 = autoencoder.layers[-3](n4)
n2 = autoencoder.layers[-2](n3)
n1 = autoencoder.layers[-1](n2)
decoder = keras.Model(decoder_input,n1)

decoded_imgs = decoder.predict(encoded_imgs)
print("Decoder Structure\n------------------------------------")
decoder.summary()

# %% [markdown]
# # Visualization

# %% [code] {"execution":{"iopub.status.busy":"2021-11-21T20:39:28.754529Z","iopub.execute_input":"2021-11-21T20:39:28.755277Z","iopub.status.idle":"2021-11-21T20:39:28.761986Z","shell.execute_reply.started":"2021-11-21T20:39:28.755238Z","shell.execute_reply":"2021-11-21T20:39:28.761264Z"}}
from random import seed
from random import random
# seed random number generator
seed(3245)
K = 10
r = int(random()*(10000-K))
r

# %% [code] {"execution":{"iopub.status.busy":"2021-11-21T20:40:12.215973Z","iopub.execute_input":"2021-11-21T20:40:12.216237Z","iopub.status.idle":"2021-11-21T20:40:16.13886Z","shell.execute_reply.started":"2021-11-21T20:40:12.216207Z","shell.execute_reply":"2021-11-21T20:40:16.138032Z"}}
import matplotlib.pyplot as plt


plt.figure(figsize=(K*5, 10))
for i in range(r, K + r):
    # Display original
    ax = plt.subplot(2, K, i-r+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, K, i - r+1  + K)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
print("Encodings 7*7*"+str(embed_filter_c))
fig, ax = plt.subplots(embed_filter_c,K,figsize=(5*K,embed_filter_c*5))
for k in range(embed_filter_c):
    for i in range(1, K + 1):
        ax[k,i-1].imshow(encoded_imgs[i+r-1,:,:,k])
        plt.gray()
        ax[k,i-1].get_xaxis().set_visible(False)
        ax[k,i-1].get_yaxis().set_visible(False)
plt.show()
