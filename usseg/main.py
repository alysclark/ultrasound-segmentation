"""Main script for running the file organisation and segmentation"""
#/usr/bin/env python3

# Python imports
import os
import sys

# Module imports
import toml

# Local imports
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)
import usseg

def main(root_dir):
    """Main function that performs all of the segmentation on a root directory"""

    # Checks and sets up the tesseract environment
    usseg.setup_tesseract()

    # Gets a list of likely ultrasound images from root dir and saves them to a pickle file.
    usseg.get_likely_us(root_dir)

    # Segments and digitises the pre-selected ultrasound images.
    usseg.segment()

    # Generates an output.html of the segmented output
    usseg.generate_html_from_pkl()

if __name__ == "__main__":
    root_dir = toml.load("config.toml")["root_dir"]
    main(root_dir)
