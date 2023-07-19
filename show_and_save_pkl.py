import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import cv2
from skimage.transform import resize
import datetime


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        # Add the folder path to the filename
        img_path = os.path.join(folder, filename)
        # Load the image in grayscale mode
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # OpenCV loads images as (height, width, channels),
            # so we need to add an extra dimension to make it (height, width, 1)
            img = img.reshape(img.shape[0], img.shape[1], 1)
            images.append(img)
    return np.array(images)


if __name__ == '__main__':
    path = 'data/'
    xname = '1_half_face_occluded.pickle'
    yname = '1_half_face_labels.pickle'
    pickle_in = open(os.path.join(path, xname), "rb")
    x = pickle.load(pickle_in)

    pickle_in = open(os.path.join(path, yname), "rb")
    y = pickle.load(pickle_in)

    print(x.shape)
    print(y.shape)

    # I use this because with 200,200 images, I exceed GPU memory.
    # TO BE CHECKED!!

    """
    x = resize(x, (len(x), 64, 64, 1), anti_aliasing=False)
    y = resize(y, (len(y), 64, 64, 1), anti_aliasing=False)

    # Print the shape after resize
    print(x.shape)
    print(y.shape)
    """

    print('load new test data')

    folder_path01 = 'data/test01_normal'
    images_normal = load_images_from_folder(folder_path01)
    folder_path02 = 'data/test01_occluded'
    images_occluded = load_images_from_folder(folder_path02)

    print(images_normal.shape)  # Should print (number of images, 224, 224, 1)
    print(images_occluded.shape)  # Should print (number of images, 224, 224, 1)

    
    # Draw the image to be sure occluded image is the same as the ground truth one
    fig = plt.figure(figsize=(6, 6))
    fig.add_subplot(1, 2, 1)
    plt.imshow(x[565, :, :, 0], cmap="gray")
    fig.add_subplot(1, 2, 2)
    plt.imshow(y[565, :, :, 0], cmap="gray")
    plt.show()
