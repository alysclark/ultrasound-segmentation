# /usr/bin/env python3

"""Segments the ultrasound images"""

# Python imports
import os
from os import walk
import logging
import sys
import re
from PIL import Image
import pickle

# Module imports
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
from pytesseract import Output
import cv2
import traceback
from skimage.measure import label, regionprops
import openpyxl
import toml

# Import sementation module
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)
from usseg import General_functions

def setup_tesseract():
    """Checks tesseract is set up appropriately

    Currently does nothing on a linux system and sets the
    pytesseract.pytesseract.tesseract_cmd to "C:/Program Files/Tesseract-OCR/tesseract.exe"
    for Windows and Cygwin systems.

    Any other system (including MACOS) a warning is displayed and nothing is done.
    It is expected, for non-Windows/Cygwin systems that tesseract is available in the PATH.

    If this is not the desired behaviour, please specify tesseract_cmd after running this
    script.

    Returns:
        tesseract_version (str) : Returns the tesseract version installed.
    """
    if sys.platform.startswith('linux'):
        pass
    elif sys.platform.startswith('win32'):
        pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
    elif sys.platform.startswith('cygwin'):
        pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
    else:
        logging.warning(
            f"Platform {sys.platform} is not recognised.\n"
            "Please ensure that you added pytesseract to your system's path."
        )

    return pytesseract.get_tesseract_version()


def segment(filenames=None, output_dir=None, pickle_path=None):
    """Segments the pre-selected ultrasound images

    Args:
        filenames (str or list, optional) : If string, must be either a single
            file name path or a path to a pickle object containing the list of
            files. Pickle objects are expected to have the extension ".pkl"
            or ".pickle".
            If a list, must be a list of filenames to ultrasound images to
            segment.
            If None, will load a test image.

        output_dir (str, optional) : Path to the output directory to store annoated
            images. If None, will load from config file.
            Defaults to None.
        pickle_path (str or bool) : If pickle_path is False, will not store the
            list of likely us images as a pickle file. If None,
            will load the pickle path from "config.toml".
            Else if a string, will dump the pickled list to the specified path.
            Defaults to None.
    """

    # filenames = next(walk('Scans/'), (None, None, []))[2]  # [] if no file
    # text_file = open("sample3_file_list_test.txt", "r")
    # filenames = text_file.read().replace('\'','').replace('[','').replace(']','').replace(' ','').split(',')
    if filenames is None:
        filenames = ["Lt_test_image.png"]

    elif isinstance(filenames, list):
        pass

    elif filenames.endswith(".pkl") or filenames.endswith(".pickle"):
        with open(filenames, "rb") as f:
            text_file = pickle.load(f)

        # Get a list of all the keys in the dictionary
        subkeys = list(text_file.keys())

        filenames = []
        # Iterate through the sublist of keys
        for key in subkeys:
            # Access the value corresponding to the key
            filenames = filenames + text_file[key]
            # print(filenames)
            #
    elif isinstance(filenames, str):
        filenames = [filenames]
    else:
        logging.warning(
            f"Unrecognised filenames type {type(filenames)}"
            "Excepted either a string or a list"
        )


    if output_dir is None:
        output_dir = toml.load("config.toml")["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    xcel_file = output_dir + "sample3_processed_data"
    Text_data = []  # text data extracted from image
    Annotated_scans = []
    Digitized_scans = []

    for input_image_filename in filenames:  # Iterare through all file names and populate excel file
        # input_image_filename = "E:/us-data-anon/0000/IHE_PDI/00003511/AA3A43F2/AAD8766D/0000371E\\EEEAE224.JPG"
        image_name = os.path.basename(input_image_filename)
        print(input_image_filename)

        try:  # Try initial segmentation
            colRGBA = Image.open(input_image_filename)  # These images are in RGBA form
            col = colRGBA.convert("RGB")  # We need RGB, so convert here.
            pix = (
                col.load()
            )  # Loads a pixel access object, where pixel values can be edited

            # from General_functions import Colour_extract, Text_from_greyscale
            COL = General_functions.Colour_extract(input_image_filename, [255, 255, 100], 80, 80)
            print("Done Colour extract")

            Fail, df = General_functions.Text_from_greyscale(input_image_filename, COL)
        except Exception:  # flat fail on 1
            traceback.print_exc()  # prints the error message and traceback
            print("Failed Text extraction")
            Text_data.append(None)
            Fail = 0
            pass

        try:  # Try initial segmentation
            segmentation_mask, Xmin, Xmax, Ymin, Ymax = General_functions.Initial_segmentation(
                input_image_filename=input_image_filename
            )
        except:  # flat fail on 1
            print("Failed Initial segmentation")
            Fail = Fail + 1
            pass

        try:  # define end ROIs
            Left_dimensions, Right_dimensions = General_functions.Define_end_ROIs(
                segmentation_mask, Xmin, Xmax, Ymin, Ymax
            )
        except:
            print("Failed Defining ROI")
            Fail = Fail + 1
            pass

        try:
            Waveform_dimensions = [Xmin, Xmax, Ymin, Ymax]
        except:
            print("Failed Waveform dimensions")
            Fail = Fail + 1
            pass

        try:  # Refine segmentation
            refined_segmentation_mask, top_curve_mask = General_functions.Segment_refinement(
                input_image_filename, Xmin, Xmax, Ymin, Ymax
            )
        except:
            traceback.print_exc()  # prints the error message and traceback
            print("Failed Segment refinement")
            Fail = Fail + 1
            pass

        try:  # Search for ticks and labels
            (
                Cs,
                ROIAX,
                CenPoints,
                onY,
                BCs,
                TYLshift,
                thresholded_image,
                Side,
                Left_dimensions,
                Right_dimensions,
                ROI2,
                ROI3,
            ) = General_functions.Search_for_ticks(
                input_image_filename,"Left", Left_dimensions, Right_dimensions
            )
            ROIAX, Lnumber, Lpositions, ROIL = General_functions.Search_for_labels(
                Cs,
                ROIAX,
                CenPoints,
                onY,
                BCs,
                TYLshift,
                Side,
                Left_dimensions,
                Right_dimensions,
                input_image_filename,
                ROI2,
                ROI3,
            )

            (
                Cs,
                ROIAX,
                CenPoints,
                onY,
                BCs,
                TYLshift,
                thresholded_image,
                Side,
                Left_dimensions,
                Right_dimensions,
                ROI2,
                ROI3,
            ) = General_functions.Search_for_ticks(
                input_image_filename,"Right", Left_dimensions, Right_dimensions
            )
            ROIAX, Rnumber, Rpositions, ROIR = General_functions.Search_for_labels(
                Cs,
                ROIAX,
                CenPoints,
                onY,
                BCs,
                TYLshift,
                Side,
                Left_dimensions,
                Right_dimensions,
                input_image_filename,
                ROI2,
                ROI3,
            )
            col = General_functions.Annotate(
                input_image_filename=input_image_filename,
                refined_segmentation_mask=refined_segmentation_mask,
                Left_dimensions=Left_dimensions,
                Right_dimensions=Right_dimensions,
                Waveform_dimensions=Waveform_dimensions,
                Left_axis=ROIL,
                Right_axis=ROIR,
            )
            Annotated_path = output_dir + image_name.partition(".")[0] + "_Annotated.png"
            fig1, ax1 = plt.subplots(1)
            ax1.imshow(col)
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.tick_params(axis="both", which="both", length=0)
            fig1.savefig(Annotated_path, dpi=900, bbox_inches="tight", pad_inches=0)
            Annotated_scans.append(Annotated_path)
        except Exception:
            traceback.print_exc()  # prints the error message and traceback
            print("Failed Axes search")
            Annotated_scans.append(None)
            Fail = Fail + 1
            pass

        try:
            Xplot, Yplot = General_functions.Plot_Digitized_data(
                Rticks=Rnumber, Rlocs=Rpositions, Lticks=Lnumber, Llocs=Lpositions
            )
            try:
                df = General_functions.Plot_correction(Xplot, Yplot, df)
                Text_data.append(df)
            except Exception:
                traceback.print_exc()
                print("Failed correction")
                continue
            Digitized_path = output_dir + image_name.partition(".")[0] + "_Digitized.png"
            plt.figure(2)
            plt.savefig(Digitized_path, dpi=900, bbox_inches="tight", pad_inches=0)
            Digitized_scans.append(Digitized_path)
        except Exception:
            print("Failed Digitization")
            traceback.print_exc()
            try:
                Text_data.append(df)
            except Exception:
                traceback.print_exc()
                Text_data.append(None)
            Digitized_scans.append(None)
            Fail = Fail + 1
            pass

        to_del = [
            "df",
            "image_name",
            "Xmax",
            "Xmin",
            "Ymax",
            "Ymin",
            "Rnumber",
            "Rpositions",
            "Lnumber",
            "Lpositions",
            "Left_dimensions",
            "Right_dimensions",
            "segmentation_mask",
        ]
        for i in to_del:
            try:
                exec("del %s" % i)
            except Exception:
                pass

        plt.close("all")
        i = 1

    print(Digitized_scans)
    print(Annotated_scans)
    print(Text_data)
    if pickle_path is not False:
        if pickle_path is None:
            pickle_path = toml.load("config.toml")["pickle"]["segmented_data"]
        with open(pickle_path, "wb") as f:
            pickle.dump([filenames, Digitized_scans, Annotated_scans, Text_data], f)
    i = 0
    return (filenames, Digitized_scans, Annotated_scans, Text_data)

if __name__ == "__main__":

    pickle_file = toml.load("config.toml")["pickle"]["likely_us_images"]
    segment(filenames=pickle_file)
