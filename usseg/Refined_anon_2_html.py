import matplotlib.pyplot as plt
import numpy as np
import pytesseract
import re
from PIL import Image
from pytesseract import Output
import cv2
import traceback
#pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
import os
import pickle
from skimage.measure import label, regionprops
from PIL import Image
import matplotlib.pyplot as plt

import openpyxl
from os import walk

# Import sementation module
import General_functions


# filenames = next(walk('Scans/'), (None, None, []))[2]  # [] if no file
# text_file = open("sample3_file_list_test.txt", "r")
# filenames = text_file.read().replace('\'','').replace('[','').replace(']','').replace(' ','').split(',')
with open("patient_paths.pkl", "rb") as f:
    text_file = pickle.load(f)

# Get a list of all the keys in the dictionary
subkeys = list(text_file.keys())

filenames = []
# Iterate through the sublist of keys
for key in subkeys:
    # Access the value corresponding to the key
    filenames = filenames + text_file[key]
    # print(filenames)


OUT_path = "/mnt/veracrypt2/us-data-processed/"
os.makedirs(OUT_path, exist_ok=True)
xcel_file = OUT_path + "sample3_processed_data"
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
    except:  # flat fail on 1
        print("Failed Initial segmentation")
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
        Annotated_path = OUT_path + image_name + "_Annotated.png"
        fig1, ax1 = plt.subplots(1)
        ax1.imshow(col)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.tick_params(axis="both", which="both", length=0)
        fig1.savefig(Annotated_path, dpi=900, bbox_inches="tight", pad_inches=0)
        Annotated_scans.append(Annotated_path)
    except:
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
        except:
            print("Failed correction")
            continue
        Digitized_path = OUT_path + image_name + "_Digitized.png"
        plt.figure(2)
        plt.savefig(Digitized_path, dpi=900, bbox_inches="tight", pad_inches=0)
        Digitized_scans.append(Digitized_path)
    except:
        print("Failed Digitization")
        try:
            Text_data.append(df)
        except:
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
        except:
            pass

    plt.close("all")
    i = 1

print(Digitized_scans)
print(Annotated_scans)
print(Text_data)
with open("lists3.pickle", "wb") as f:
    pickle.dump([filenames, Digitized_scans, Annotated_scans, Text_data], f)
i = 0
