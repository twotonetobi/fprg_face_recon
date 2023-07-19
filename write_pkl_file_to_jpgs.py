import os
import pickle
import cv2

# Load pickle file
with open('data/pkl/pred_all_pkl.pkl', 'rb') as f:
    images = pickle.load(f)

# Define your output folder
output_folder = 'results_affectnet'

# Check if output_folder exists, if not, create it
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through each image in the numpy array
for i, img in enumerate(images):
    # Construct the filename using the index of the image in the array.
    # The str.zfill method is used to pad the filename with leading zeros.
    filename = str(i + 1).zfill(6) + '.jpg'
    filepath = os.path.join(output_folder, filename)
    # The image array is normalized to 0-255 and converted to 'uint8'
    # before saving, because cv2.imwrite expects byte values.
    # cv2.imwrite(filepath, (img * 255).astype('uint8'))
    cv2.imwrite(filepath, (img * 255).astype('uint8'), [cv2.IMWRITE_JPEG_QUALITY, 96])

