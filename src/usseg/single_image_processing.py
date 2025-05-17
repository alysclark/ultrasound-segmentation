"""Segment a single ultrasound image object.

This module provides a number of functions for segmenting single ultrasound images,
extracting segmentation and textual data from the images.

**Usage:**

To segment a single ultrasound image, you can use the following code:

```python
from usseg.Single_image_processing import data_from_image

# Load the ultrasound image.
PIL_img = ...
cv2_img = ...

# Extract segmentation and textual data from the image.
df, XYdata = data_from_image(PIL_img, cv2_img)```
"""
# Python imports
import os
import sys
import logging

# Module imports
import matplotlib.pyplot as plt

# Local imports
from usseg import general_functions

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
    # Extracts yellow text from image
    #PIL_img , cv2_img = General_functions.upscale_both_images(PIL_img,cv2_img)
    PIL_image_RGB = PIL_img.convert("RGB")  # We need RGB, so convert here. with PIL
    COL = General_functions.colour_extract_vectorized(PIL_image_RGB, [255, 255, 100], 95, 95)

    #COL = General_functions.Colour_extract(PIL_image_RGB, [255, 255, 100], 100, 100)
    text_extract_failed, df = General_functions.text_from_greyscale(cv2_img, COL)
    # Failure not really relavent to the rest of the segmenation so just logged as 
    # a warning for the end user.
    if text_extract_failed:
        logger.warning("Couldn't extract text from image. Continuing...")
    else:
        logger.info("Completed colour extraction.")

    # No error handling for initial segmentation as impossible to complete segmentation
    # without segmenation mask.
    segmentation_mask, Xmin, Xmax, Ymin, Ymax = General_functions.initial_segmentation(
        input_image_obj=PIL_image_RGB
    )

    # Gets ROIS
    Left_dimensions, Right_dimensions = General_functions.define_end_rois(
        segmentation_mask, Xmin, Xmax, Ymin, Ymax
    )

    # Search for ticks and labels
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
    ) = General_functions.search_for_ticks(
        cv2_img, "Left", Left_dimensions, Right_dimensions
    )
    ROIAX, Lnumber, Lpositions, ROIL = General_functions.search_for_labels(
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
    ) = General_functions.search_for_ticks(
        cv2_img, "Right", Left_dimensions, Right_dimensions
    )
    ROIAX, Rnumber, Rpositions, ROIR = General_functions.search_for_labels(
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
        refined_segmentation_mask, top_curve_mask, top_curve_coords
    ) = General_functions.segment_refinement(
        cv2_img, Xmin, Xmax, Ymin, Ymax
    )

    # Gets the segmentation
    Xplot, Yplot, Ynought = General_functions.plot_digitized_data(
        Rnumber, Rpositions, Lnumber, Lpositions, top_curve_coords,
    )

    if not text_extract_failed:
        df = General_functions.plot_correction(Xplot, Yplot, df)

    plt.close("all")
    XYdata = [Xplot, Yplot]
    return df, XYdata
