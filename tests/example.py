# Python imports
import os

# Local imports
import usseg

from PIL import Image
import cv2
import matplotlib.pyplot as plt


def main(pil_img, cv2_img):
    """Main function that performs all the segmentation on a root directory"""
    # Checks and sets up the tesseract environment
    usseg.setup_tesseract()

    # Process a single image
    textual_data, raw_signal = usseg.data_from_image(pil_img, cv2_img)
    print(textual_data)
    plt.figure()
    plt.plot(raw_signal[0], raw_signal[1], "-")


if __name__ == "__main__":
    filename = os.path.abspath("Lt_test_image.png")
    pil_img_read = Image.open(filename)
    cv2_img_read = cv2.imread(filename)
    main(pil_img_read, cv2_img_read)
