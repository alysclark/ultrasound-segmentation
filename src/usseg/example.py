# Python imports
import os
import sys

# Local imports
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)
import usseg

from PIL import Image
import cv2
import matplotlib.pyplot as plt

def main(PIL_img, cv2_img):
    """Main function that performs all of the segmentation on a root directory"""
    # Checks and sets up the tesseract environment
    usseg.setup_tesseract()

    # Process a single image
    textual_data, raw_signal = usseg.data_from_image(PIL_img, cv2_img)
    print(textual_data)
    plt.figure()
    plt.plot(raw_signal[0], raw_signal[1], "-")

if __name__ == "__main__":
    filename = os.path.abspath("Lt_test_image.png")
    PIL_img = Image.open(filename)
    cv2_img = cv2.imread(filename)
    main(PIL_img, cv2_img)
