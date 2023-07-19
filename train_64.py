import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import cv2

import keras
import tensorflow as tf
from keras.backend import set_session

from skimage.transform import resize

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout, LeakyReLU, Conv2DTranspose, ReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Reshape
from tensorflow.keras import layers
import datetime
from keras import initializers

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9  # fraction of memory
config.gpu_options.visible_device_list = "0"

set_session(tf.Session(config=config))


# Here is the function to create a generator. 
def creategen():
    generator = Sequential()

    generator.add(
        Conv2D(64, (5, 5), strides=(2, 2), input_shape=x.shape[1:], padding="SAME", kernel_initializer='random_normal'))
    generator.add(BatchNormalization())
    generator.add(ReLU())
    generator.add(Dropout(0.3))

    generator.add(Conv2D(128, (5, 5), strides=(2, 2), padding="SAME", kernel_initializer='random_normal'))
    generator.add(BatchNormalization())
    generator.add(ReLU())
    generator.add(Dropout(0.3))

    generator.add(Conv2D(256, (5, 5), strides=(2, 2), padding="SAME", kernel_initializer='random_normal'))
    generator.add(BatchNormalization())
    generator.add(ReLU())
    generator.add(Dropout(0.3))

    generator.add(Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    generator.add(BatchNormalization())
    generator.add(ReLU())

    generator.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    generator.add(BatchNormalization())
    generator.add(ReLU())

    generator.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation="tanh"))

    return generator


# Here is the function to create a discriminator.
def createdisc():
    discriminator = Sequential()

    discriminator.add(
        Conv2D(64, (5, 5), strides=(2, 2), input_shape=x.shape[1:], padding="SAME", kernel_initializer='random_normal'))
    discriminator.add(BatchNormalization())
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Conv2D(128, (5, 5), strides=(2, 2), padding="SAME", kernel_initializer='random_normal'))
    discriminator.add(BatchNormalization())
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Conv2D(256, (5, 5), strides=(2, 2), padding="SAME", kernel_initializer='random_normal'))
    discriminator.add(BatchNormalization())
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Flatten())
    discriminator.add(Dense(1))

    return discriminator

# Here is the function to create the GAN out of the generator and the discrimator.
def creategan(generator, discriminator):
    gan = Sequential()
    gan.add(generator)
    discriminator.trainable = False
    gan.add(discriminator)
    return (gan)


# This is to save models. 
def save_models(gan, discriminator, generator, path, epoch):
    datenow = str(datetime.datetime.now().strftime('%m-%d-%H:%M'))
    gan.save_weights(os.path.join(path, "{0}_wgan_{1}.h5".format(epoch, datenow)))
    gan.save(os.path.join(path, "{0}_mgan_{1}.h5".format(epoch, datenow)))

    discriminator.save_weights(os.path.join(path, "{0}_wd_{1}.h5".format(epoch, datenow)))
    discriminator.save(os.path.join(path, "{0}_md_{1}.h5".format(epoch, datenow)))

    generator.save_weights(os.path.join(path, "{0}_wg_{1}.h5".format(epoch, datenow)))
    generator.save(os.path.join(path, "{0}_mg_{1}.h5".format(epoch, datenow)))


# This is to obtain test losses in each training step. 
def test(x, y):
    gan_inp_t = x
    gan_label_t = np.ones([len(gan_inp_t)])
    gan_predict_t = None

    disc_inp_t = None
    disc_label_t = np.zeros([len(gan_inp_t) * 2])
    disc_label_t[len(gan_inp_t):] = 1
    disc_predict_t = None

    gen_predict_t = generator.predict(gan_inp_t)

    disc_inp_t = np.concatenate((gen_predict_t, y), axis=0)
    disc_predict_t = discriminator.predict(disc_inp_t)

    d_loss_t = discriminator.test_on_batch(disc_inp_t, disc_label_t)
    gan_loss_t = gan.test_on_batch(gan_inp_t, gan_label_t)

    return (gan_loss_t, d_loss_t)


# I also use smooth/noisy labels proposed by Salimans et al 2016
# Reference: https://github.com/soumith/ganhacks
def train(x, y, nepoch, model_save_path='checkpoints'):
    gen_predict = None
    # Initialize the inputs and the labels.
    gan_inp = x
    gan_label = np.ones(64)
    gan_predict = None

    disc_inp = None

    disc_label = np.zeros(64 * 2)
    disc_label[64:] = 1
    disc_predict = None
    sess = tf.Session()

    for epoch in range(nepoch):

        for batch_ctr in range(65):

            # Generator makes a prediction.
            gen_predict = generator.predict(gan_inp[batch_ctr * 64:(batch_ctr + 1) * 64])

            # Minibatch isolation and label smoothing is done here:
            if (epoch % 2 == 0):
                disc_inp = gen_predict
                disc_label = np.random.normal(loc=0, scale=0.10, size=64)
            else:
                disc_inp = y[batch_ctr * 64:(batch_ctr + 1) * 64]
                disc_label = np.random.normal(loc=1, scale=0.10, size=64)

            # Initialize a label variable for generator to use it in training.
            gen_label = y[batch_ctr * 64:(batch_ctr + 1) * 64]

            # Do one training step. Also assign the losses to variables. 
            d_loss = discriminator.train_on_batch(disc_inp, disc_label)
            gan_loss = gan.train_on_batch(gan_inp[batch_ctr * 64:(batch_ctr + 1) * 64],
                                          gan_label)
            gen_loss = generator.train_on_batch(gan_inp[batch_ctr * 64:(batch_ctr + 1) * 64], gen_label)

        if (epoch + 1) % 500 == 0:
            save_models(gan, discriminator, generator, model_save_path, epoch + 1)
            print("MODEL SAVED")

        # Test images are the images after the 240000 image. It makes 11261 test images.
        (tgan, tdisc) = test(x[250000:], y[250000:])
        print("Epoch: {2} Gan Loss: {0}       Disc Loss: {1}        Gen Loss: {3}".format(gan_loss, d_loss, epoch + 1,
                                                                                          gen_loss))
        print("Epoch: {2} Test Gan Loss: {0}  Test Disc Loss: {1} \n\n\n".format(tgan, tdisc, epoch + 1))


def count_images(directory):
    return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and (f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.gif'))])


def load_images_from_folder(folder):
    num_images = count_images(folder)
    # Initialize an empty numpy array of the required size
    images = np.zeros((num_images, 224, 224, 1), dtype='uint8')  # using uint8 to save memory

    for filename in os.listdir(folder):
        # Extract the image number from the filename
        image_number = int(filename.split('.')[0]) - 1 
        # Add the folder path to the filename
        img_path = os.path.join(folder, filename)
        # Load the image in grayscale mode
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is not None:
            # OpenCV loads images as (height, width, channels),
            # so we need to add an extra dimension to make it (height, width, 1)
            img = img.reshape(img.shape[0], img.shape[1], 1)
            # Use the image number to index into the images array
            images[image_number] = img

    return images



if __name__ == '__main__':

    ########
    ########
    # Let this part run for training #
    ########
    ########

    folder_path01 = 'data/normal'
    images_normal = load_images_from_folder(folder_path01)
    folder_path02 = 'data/occluded'
    images_occluded = load_images_from_folder(folder_path02)

    print(images_normal.shape)  # Should print (number of images, 224, 224, 1)
    print(images_occluded.shape)  # Should print (number of images, 224, 224, 1)

    # adopt
    pickle_path = 'pkl'

    print(images_normal.shape)
    images_resized_normal = []
    
    # Draw the image to be sure occluded image is the same as the ground truth one
    fig = plt.figure(figsize=(6, 6))
    fig.add_subplot(1, 2, 1)
    plt.imshow(images_occluded[3, :, :, 0], cmap="gray")
    fig.add_subplot(1, 2, 2)
    plt.imshow(images_normal[3, :, :, 0], cmap="gray")
    plt.show()

    # resize all the images
    for i in range(len(images_normal)):
        img = images_normal[i]
        new_img = resize(img, (64, 64, 1), anti_aliasing=False)
        images_resized_normal.append(new_img)
        # If you don't need the original image anymore, you can delete it to free memory
        images_normal[i] = 1
        if (i + 1) % 100 == 0:
            print(f"Processed 1st {i + 1}  normal images.")

    # Convert list back to numpy array
    images_resized_normal = np.array(images_resized_normal)
    images_resized_normal1_2 = images_resized_normal.astype(np.float32)
    # Print the shape after resize
    print(images_resized_normal1_2.shape)
    print(images_resized_normal1_2.dtype)
    print(images_resized_normal1_2[50, 50, 50, 0])

    # When handling large image numbers, it make sense to load the resized pkl files 
    # and do the rescaling as an extra step
    # Save images_normal
    with open(pickle_path + 'images_resized_normal1_2.pkl', 'wb') as f:
        pickle.dump(images_resized_normal1_2, f)

    y = images_resized_normal1_2


    images_resized_occluded = []
    for i in range(len(images_normal)):
        img = images_occluded[i]
        new_img = resize(img, (64, 64, 1), anti_aliasing=False)
        images_resized_occluded.append(new_img)
        # If you don't need the original image anymore, you can delete it to free memory
        images_occluded[i] = 1
        if (i + 1) % 100 == 0:
            print(f"Processed 1st {i + 1}  occluded images.")

    # Convert list back to numpy array
    images_resized_occluded1_2 = np.array(images_resized_occluded)
    images_resized_occluded1_2 = images_resized_occluded1_2.astype(np.float32)
    images_resized_occluded = []
    # Print the shape after resize
    print(images_resized_occluded1_2.shape)
    print(images_resized_occluded1_2.dtype)
    print(images_resized_occluded1_2[50, 50, 50, 0])

    # When handling large image numbers, it make sense to load the resized pkl files 
    # and do the rescaling as an extra step
    # Save images_occluded
    with open(pickle_path + 'images_resized_occluded1_2.pkl', 'wb') as f:
        pickle.dump(images_resized_occluded1_2, f)

    x = images_resized_occluded1_2

    ########
    ########
    # Generate the GAN #
    ########
    ########

    generator = creategen()
    discriminator = createdisc()
    generator.summary()

    opt_disc = Adam(lr=0.00004)
    discriminator.trainable = True
    discriminator.compile(loss="binary_crossentropy", optimizer=opt_disc)
    discriminator.summary()

    gan = creategan(generator, discriminator)

    opt_gan = Adam(lr=0.00001)
    gan.compile(loss="binary_crossentropy", optimizer=opt_gan)
    gan.summary()

    # Checking lengths of the input and ground truth arrays. Also checking if normalization is done.
    print(len(x), len(y))
    print(x.max(), x.min())
    print(y.max(), y.min())
    #
    # Compile
    opt_disc = Adam(lr=0.00004)
    discriminator.trainable = True
    discriminator.compile(loss="binary_crossentropy", optimizer=opt_disc)
    discriminator.summary()

    opt_gen = Adam(lr=0.00001)
    generator.compile(loss='mean_squared_error', optimizer=opt_gen)
    generator.summary()

    opt_gan = Adam(lr=0.00001)
    gan.compile(loss="binary_crossentropy", optimizer=opt_gan)
    gan.summary()

    ########
    ########
    # Start Training Loop #
    ########
    ########
    batch_size = 64
    train(x, y, 6004)


    print('finished predict')