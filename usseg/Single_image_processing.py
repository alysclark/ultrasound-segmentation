# /usr/bin/env python3

"""Segments a single ultrasound image object"""

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

# Local imports
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)
from usseg import General_functions

def data_from_image(PIL_img,cv2_img):
    Text_data = []  # text data extracted from image
    Annotated_scans = []
    Digitized_scans = []
    try:  # Try text extraction
        PIL_image_RGB = PIL_img.convert("RGB")  # We need RGB, so convert here. with PIL

        # from General_functions import Colour_extract, Text_from_greyscale
        COL = General_functions.Colour_extract(PIL_image_RGB, [255, 255, 100], 80, 80)
        print("Done Colour extract")

        Fail, df = General_functions.Text_from_greyscale(cv2_img, COL)
    except Exception:  # flat fail on 1
        traceback.print_exc()  # prints the error message and traceback
        print("Failed Text extraction")
        Text_data.append(None)
        Fail = 0
        pass

    try:  # Try initial segmentation
        segmentation_mask, Xmin, Xmax, Ymin, Ymax = General_functions.Initial_segmentation(
            input_image_obj=PIL_image_RGB
        )
    except Exception:  # flat fail on 1
        print("Failed Initial segmentation")
        Fail = Fail + 1
        pass

    try:  # define end ROIs
        Left_dimensions, Right_dimensions = General_functions.Define_end_ROIs(
            segmentation_mask, Xmin, Xmax, Ymin, Ymax
        )
    except Exception:
        print("Failed Defining ROI")
        Fail = Fail + 1
        pass

    try:
        Waveform_dimensions = [Xmin, Xmax, Ymin, Ymax]
    except Exception:
        print("Failed Waveform dimensions")
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
            cv2_img, "Left", Left_dimensions, Right_dimensions
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
            cv2_img,
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
            cv2_img, "Right", Left_dimensions, Right_dimensions
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
            cv2_img,
            ROI2,
            ROI3,
        )
    except Exception:
        traceback.print_exc()  # prints the error message and traceback
        print("Failed Axes search")
        
        Fail = Fail + 1
        pass

    try:
        Xplot, Yplot, Ynought = General_functions.Plot_Digitized_data(
            Rticks=Rnumber, Rlocs=Rpositions, Lticks=Lnumber, Llocs=Lpositions
        )
        
        try:  # Refine segmentation
            refined_segmentation_mask, top_curve_mask = General_functions.Segment_refinement(
                cv2_img, Xmin, Xmax, Ymin, Ymax
            )
        except Exception:
            traceback.print_exc()  # prints the error message and traceback
            print("Failed Segment refinement")
            Fail = Fail + 1
            pass

        try:
            df = General_functions.Plot_correction(Xplot, Yplot, df)
            Text_data.append(df)
        except Exception:
            traceback.print_exc()
            print("Failed correction")
            pass
    except Exception:
        print("Failed Digitization")
        traceback.print_exc()
        try:
            Text_data.append(df)
        except Exception:
            traceback.print_exc()
            Text_data.append(None)
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
    XYdata = [Xplot,Yplot]
    return df, XYdata
