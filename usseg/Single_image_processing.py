# /usr/bin/env python3

"""Segments a single ultrasound image object"""

# Python imports
import os
import sys
import logging

# Module imports
import matplotlib.pyplot as plt
import traceback

# Local imports
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)
from usseg import General_functions

logger = logging.getLogger(__file__)

def data_from_image(PIL_img,cv2_img):
    """Extract segmentation and textual data from an image.

    Args:
        PIL_img (Pillow Image object) : The image in Pillow format.
        cv2_img (cv2 Image object) : The image in cv2 format.

    Returns:
        df (pandas dataframe) : Dataframe of extracted text.
        XYdata (list) : X and Y coordinates of the extracted segmentation.
    """
    Text_data = []  # text data extracted from image
    try:  # Try text extraction
        PIL_image_RGB = PIL_img.convert("RGB")  # We need RGB, so convert here. with PIL

        # from General_functions import Colour_extract, Text_from_greyscale
        COL = General_functions.Colour_extract(PIL_image_RGB, [255, 255, 100], 80, 80)
        logger.info("Completed colour extraction")

        Fail, df = General_functions.Text_from_greyscale(cv2_img, COL)
    except Exception:  # flat fail on 1
        traceback.print_exc()  # prints the error message and traceback
        logger.error("Failed colour or text extraction")
        Text_data.append(None)
        Fail = 0
        pass

    try:  # Try initial segmentation
        segmentation_mask, Xmin, Xmax, Ymin, Ymax = General_functions.Initial_segmentation(
            input_image_obj=PIL_image_RGB
        )
    except Exception:  # flat fail on 1
        logger.error("Failed Initial segmentation")
        Fail = Fail + 1
        pass

    try:  # define end ROIs
        Left_dimensions, Right_dimensions = General_functions.Define_end_ROIs(
            segmentation_mask, Xmin, Xmax, Ymin, Ymax
        )
    except Exception:
        logger.error("Failed Defining ROI")
        Fail = Fail + 1
        pass

    try:
        Waveform_dimensions = [Xmin, Xmax, Ymin, Ymax]
    except Exception:
        logger.error("Failed Waveform dimensions")
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
        logger.error("Failed Axes search")
        
        Fail = Fail + 1
        pass

    try:  # Refine segmentation
        (
            refined_segmentation_mask, top_curve_mask, top_curve_coords
        ) = General_functions.Segment_refinement(
            cv2_img, Xmin, Xmax, Ymin, Ymax
        )
    except Exception:
        traceback.print_exc()  # prints the error message and traceback
        logger.error("Failed Segment refinement")
        Fail = Fail + 1
        pass

    try: 
        Xplot, Yplot, Ynought = General_functions.Plot_Digitized_data(
            Rnumber, Rpositions, Lnumber, Lpositions, top_curve_coords,
        )

    except Exception:
        logger.error("Failed Digitization")
        traceback.print_exc()
        try:
            Text_data.append(df)
        except Exception:
            traceback.print_exc()
            Text_data.append(None)
        Fail = Fail + 1
        pass

    try:
        df = General_functions.Plot_correction(Xplot, Yplot, df)
        Text_data.append(df)
    except Exception:
        traceback.print_exc()
        logger.error("Failed correction")
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
