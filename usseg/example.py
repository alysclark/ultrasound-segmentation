# Python imports
import os
import sys

# Module imports
import toml

# Local imports
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)
import usseg

from PIL import Image
import cv2
import matplotlib.pyplot as plt
PIL_img = Image.open('E:\\us-data-anon\\0000\\IHE_PDI\\00003511\\AA3A43F2\\AAD8766D\\0000371E\\EE511F45.JPG')
cv2_img = cv2.imread('E:\\us-data-anon\\0000\\IHE_PDI\\00003511\\AA3A43F2\\AAD8766D\\0000371E\\EE511F45.JPG')

def main(root_dir):
    """Main function that performs all of the segmentation on a root directory"""
    # Checks and sets up the tesseract environment
    usseg.setup_tesseract()

    # Process a single image
    textual_data, raw_signal = usseg.data_from_image(PIL_img, cv2_img)
    print(textual_data)
    plt.figure()
    plt.plot(raw_signal[0], raw_signal[1], "-")

i=0
if __name__ == "__main__":
    root_dir = toml.load("config.toml")["root_dir"]
    main(root_dir)