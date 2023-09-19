""" A set of functions to segment and extract data from doppler ultrasound scans"""
# Python imports
import traceback
import math
import re
import logging
from functools import lru_cache

# Module imports
import matplotlib.pyplot as plt
from skimage import morphology, measure
import numpy as np
from skimage.measure import find_contours
from skimage.draw import polygon_perimeter
import cv2
from PIL import Image
import scipy
from scipy.ndimage.filters import gaussian_filter
from scipy import signal
from scipy.spatial.distance import cdist
from scipy.signal import find_peaks, peak_widths
import statistics
import scipy.linalg
from sklearn.cluster import DBSCAN
import pandas as pd
import pytesseract
from pytesseract import Output


logger = logging.getLogger(__file__)

class _HashableArray(np.ndarray):
    def __new__(self, *args, **kwargs):
        obj = np.asarray(*args, **kwargs).view(self)
        return obj

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        if self.size > 1:
            return hash(tuple(self.view()))
        return hash(self.view().item)


def Initial_segmentation(input_image_obj):
    """Perform an initial corse segmentation of the waveform.

    Args:
        input_image_filename (str) : Name of file within current directory, or path to a file.

    Returns:
        segmentation_mask (ndarray) : A binary array mask showing the corse segmentation of waveform (1) against background (0).
        Xmin (float) : Minimum X coordinate of the segmentation.
        Xmax (float) : Maximum X coordinate of the segmentation.
        Ymin (float) : Minimum Y coordinate of the segmentation.
        Ymax (float) : Maximum Y coordinate of the segmentation.
    """
    # img_RGBA = Image.open(input_image_filename)  # These images are in RGBA form
    img_RGB = input_image_obj
    pixel_data = img_RGB.load()  # Loads a pixel access object, where pixel values can be edited
    # gry = img_RGB.convert("L")  # (returns grayscale version)

    # To threshold the ROI,
    for y in range(img_RGB.size[1]):
        for x in range(img_RGB.size[0]):
            r = pixel_data[x, y][0]  # red component
            g = pixel_data[x, y][1]  # green component
            b = pixel_data[x, y][2]  # blue component
            rgb_values = [r, g, b]
            min_rgb = min(rgb_values)
            max_rgb = max(rgb_values)
            rgb_range = max_rgb - min_rgb  # range across RGB components

            # notice that the spread of values across R, G and B is reasonably small as the colours is a shade of white/grey,
            # It can be isolated by marking pixels with a low range (<50) and resonable brightness (sum or R G B components > 120)
            if (rgb_range < 100 and max_rgb > 120):  # NEEDS REFINING - these values seem to be optimal for the cases i've tested.
                pixel_data[x, y] = (
                    255,
                    255,
                    255)  # mark all pixels meeting the conditions to white.
            else:
                pixel_data[x, y] = (0, 0, 0)  # If conditions not met, set to black/

            if img_RGB.size[1] > 600:
                if y < 400:  # for some reason x==0 is white, this line negates this.
                    pixel_data[x, y] = (0, 0, 0)
            else:
                if y < 20:  # for some reason x==0 is white, this line negates this.
                    pixel_data[x, y] = (0, 0, 0)

    binary_image = np.asarray(img_RGB)  # Make the image an nparray
    pixel_sum = binary_image.sum(-1)  # sum over each pixel (255,255,255)->[765]
    nonzero_pixels = (pixel_sum > 0).astype(bool)  # Change type
    # Some processing to refine the target area
    segmentation_mask = morphology.remove_small_objects(
        nonzero_pixels, 200, connectivity=2
    )  # Remover small objects (noise)
    segmentation_mask = morphology.remove_small_holes(segmentation_mask, 200)  # Fill in any small holes
    segmentation_mask = morphology.binary_erosion(
        segmentation_mask
    )  # Erode the remaining binary, this can remove any ticks that may be joined to the main body
    segmentation_mask = morphology.binary_erosion(segmentation_mask)  # Same as above - combine to one line if possible
    segmentation_mask = morphology.binary_dilation(
        segmentation_mask
    )  # Dilate to try and recover some of the collateral loss through erosion
    segmentation_mask = segmentation_mask.astype(float)  # Change type

    contours = find_contours(segmentation_mask)  # contours of each object withing segmentation_mask

    xmin_list, xmax_list, ymin_list, ymax_list = (
        [],
        [],
        [],
        [],
    )  # initialise some variables to store max and min values

    for contour in contours:  # find max and mins of each contour
        xmin_list.append(np.min(contour[:, 1]))
        xmax_list.append(np.max(contour[:, 1]))
        ymin_list.append(np.min(contour[:, 0]))
        ymax_list.append(np.max(contour[:, 0]))

    Xmin, Xmax, Ymin, Ymax = (
        np.min(xmin_list),
        np.max(xmax_list),
        np.min(ymin_list),
        np.max(ymax_list),
    )  # find max and min withing the lists.

    return segmentation_mask, Xmin, Xmax, Ymin, Ymax

def Define_end_ROIs(segmentation_mask, Xmin, Xmax, Ymin, Ymax):
    """
    Function to define regions adjacent to the corse waveform in which to search for axes info.

    Args:
        segmentation_mask (ndarray) : A binary array mask showing the corse segmentation of waveform (1) against background (0).
        Xmin (float) : Minimum X coordinate of the segmentation.
        Xmax (float) : Maximum X coordinate of the segmentation.
        Ymin (float) : Minimum Y coordinate of the segmentation.
        Ymax (float) : Maximum Y coordinate of the segmentation.

    Returns:
        Left_dimensions (list) : edge points for the left axes ROI [Xmin, Xmax, Ymin, Ymax].
        Right_dimensions (list) : edge points for the left axes ROI [Xmin, Xmax, Ymin, Ymax].
    """

    # For defining the specific ROI either side of the waveform data
    # these ROIs are later used to search for ticks and labels

    Ylim, Xlim = segmentation_mask.shape[0], segmentation_mask.shape[1]  # segmentation_mask is shaped as [y,x] fyi.
    # LHS
    Xmin_L = 0  # Xmin - 50
    Xmax_L = Xmin - 1
    if (Ymin - 125) > 0:
        Ymin_L = Ymin - 25
    else:
        Ymin_L = 1
    Ymax_L = Ylim
    Left_dimensions = [Xmin_L, Xmax_L, Ymin_L, Ymax_L]
    # # RHS
    Xmin_R = Xmax
    Xmax_R = Xlim  # Xmax + 50
    if (Ymin - 125) > 0:
        Ymin_R = Ymin - 25
    else:
        Ymin_R = 1

    if (Ylim - 70) > 0:
        Ymax_R = Ylim - 70
    else:
        Ymax_R = Ylim

    Right_dimensions = [Xmin_R, Xmax_R, Ymin_R, Ymax_R]
    return Left_dimensions, Right_dimensions

def check_inverted_curve(top_curve_mask, Ymax, Ymin, tol=.25):
    """Checks to see if top curve mask is of an inverted waveform

    Args:
        top_curve_mask (ndarray) : A binary array showing a curve along the top of the refined waveform.
        Ymax (float) : Maximum Y coordinate of the segmentation in pixels.
        Ymin (float) : Minimum Y coordinate of the segmentation in pixels.
        tol (float, optional) : If the top curve occupies less than tol * (Ymax - Ymin) rows, then
            the curve is assumed to be inverted (that is True is returned). If the top curve occupies more than
            or equal to this number of rows, the False is returned and the curve is assumed to be non-inverted.
            Defaults to 0.45.

    Returns:
        return value (bool) : True if the top curve is of an inverted waveform, False is the top curve is of a non-inverted waveform.
    """
    c_rows = np.where(np.sum(top_curve_mask, axis=1))   # Curve rows
    c_range = np.max(c_rows) - np.min(c_rows)           # Y range of top curve
    return c_range / (Ymax - Ymin) < tol

def Segment_refinement(input_image_obj, Xmin, Xmax, Ymin, Ymax):
    """
    Function to refine the waveform segmentation within the bounds of the corse waveform ROI.

    Args:
        input_image_filename (str) : Name of file within current directory, or path to a file.
        Xmin (float) : Minimum X coordinate of the segmentation in pixels.
        Xmax (float) : Maximum X coordinate of the segmentation in pixels.
        Ymin (float) : Minimum Y coordinate of the segmentation in pixels.
        Ymax (float) : Maximum Y coordinate of the segmentation in pixels.
    Returns:
        refined_segmentation_mask (ndarray) :  A binary array mask showing the refined segmentation of waveform (1) against background (0).
        top_curve_mask (ndarray) : A binary array showing a curve along the top of the refined waveform.
    	top_curve_coords (ndarray) : A list of the coordinates for the top curve.
    """

    # Refine segmentation to increase smoothing
    # Save output to .txt file to load later.

    image = input_image_obj
    input_image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresholded_image = cv2.threshold(input_image_gray, 30, 255, 0)
    thresholded_image[:, int(Xmax) : -1] = 0
    thresholded_image[:, 0 : int(Xmin) - 1] = 0
    thresholded_image[0 : int(Ymin) - 50, :] = 0
    thresholded_image[int(Ymax) : -1, :] = 0
    main_ROI = thresholded_image  # Main ROI

    binary_image = main_ROI  # Make the image an nparray
    nonzero_pixels = (binary_image > 0).astype(bool)  # Change type
    # Some processing to refine the target area
    refined_segmentation_mask = morphology.remove_small_objects(
        nonzero_pixels, 200, connectivity=2
    )  # Remover small objects (noise)
    refined_segmentation_mask = morphology.remove_small_holes(refined_segmentation_mask, 200)  # Fill in any small holes
    refined_segmentation_mask = morphology.binary_erosion(
        refined_segmentation_mask
    )  # Erode the remaining binary, this can remove any ticks that may be joined to the main body
    refined_segmentation_mask = morphology.binary_erosion(refined_segmentation_mask)  # Same as above - combine to one line if possible
    refined_segmentation_mask = morphology.binary_dilation(
        refined_segmentation_mask
    )  # Dilate to try and recover some of the collateral loss through erosion
    refined_segmentation_mask = refined_segmentation_mask.astype(float)  # Change type

    refined_segmentation_mask = morphology.binary_dilation(refined_segmentation_mask)
    refined_segmentation_mask = morphology.remove_small_holes(refined_segmentation_mask, 1000)
    refined_segmentation_mask = morphology.closing(refined_segmentation_mask)
    refined_segmentation_mask = refined_segmentation_mask.astype(int)
    refined_segmentation_mask = scipy.signal.medfilt(refined_segmentation_mask, 3)

    # assuming mask is a binary image
    # label and calculate parameters for every cluster in mask
    labelled = measure.label(refined_segmentation_mask)
    rp = measure.regionprops(labelled)

    # get size of largest cluster

    sizes = sorted([i.area for i in rp])
    refined_segmentation_mask = refined_segmentation_mask.astype(bool)
    # remove everything smaller than largest
    try:
        refined_segmentation_mask = morphology.remove_small_objects(refined_segmentation_mask, min_size=sizes[-2] + 10)
    except Exception:
        pass
    refined_segmentation_mask = refined_segmentation_mask.astype(float)
    # refined_segmentation_mask[rr, cc] = 1 #set color white

    blurred = gaussian_filter(refined_segmentation_mask, sigma=7)
    refined_segmentation_mask = (blurred > 0.5) * 1
    labelled = measure.label(refined_segmentation_mask)
    rp = measure.regionprops(labelled)

    ws = morphology.binary_erosion(refined_segmentation_mask)
    ws = ws.astype(float)
    top_curve_mask = refined_segmentation_mask - ws
    for x in range(int(rp[0].centroid[0]), top_curve_mask.shape[0]):
        top_curve_mask[x, :] = 0

    # Checks if waveforms are inverted, if so gets the bottom of the curve
    if check_inverted_curve(top_curve_mask, Ymax, Ymin):
        top_curve_mask = refined_segmentation_mask - ws
        for x in range(0, int(rp[0].centroid[0])):
            top_curve_mask[x, :] = 0

    top_curve_coords = np.array(list(zip(*np.nonzero(top_curve_mask))))

    return refined_segmentation_mask, top_curve_mask, top_curve_coords

def Search_for_ticks(input_image_obj, Side, Left_dimensions, Right_dimensions):
    """
    Function to search for the ticks in either of the axes ROIs

    Args:
        input_image_filename (str) : Name of file within current directory, or path to a file.
        Side (str) : Indicates the 'Left' or 'Right' axes.
        Left_dimensions (list) : edge points for the left axes ROI [Xmin, Xmax, Ymin, Ymax].
        Right_dimensions (list) : edge points for the left axes ROI [Xmin, Xmax, Ymin, Ymax].

    Returns:
        Cs (tuple) : list of contours for ticks found.
        ROIAX (ndarray) : narray defining the ROI to search for the axes.
        CenPoints (list) : center points for the ticks identified.
        onY (list) : indexes of the contours which lie on the target x plane.
        BCs (list) : Contours of the objects which lie on the target x plane.
        TYLshift (intc) : shift in the x coordninate bounds - reducing the axes ROI in which to search for axes text.
        thresholded_image (ndarray) : Threshold values iterated through.
        Side (str) : Indicates the 'Left' or 'Right' axes.
        Left_dimensions (list) : edge points for the left axes ROI [Xmin, Xmax, Ymin, Ymax].
        Right_dimensions (list) : edge points for the left axes ROI [Xmin, Xmax, Ymin, Ymax].
        ROI2
        ROI3
    """

    image = input_image_obj
    thresholded_image = image

    if Side == "Left":
        ROIAX = thresholded_image[
            int(Left_dimensions[2]) : int(Left_dimensions[3]),
            int(Left_dimensions[0]) : int(Left_dimensions[1]),
        ]  # Right ROI
    elif Side == "Right":
        ROIAX = thresholded_image[
            int(Right_dimensions[2]) : int(Right_dimensions[3]),
            int(Right_dimensions[0]) : int(Right_dimensions[1]),
        ]  # Left ROI

    RGBnp = np.array(ROIAX)  # convert images to array (not sure needed)
    RGBnp[RGBnp <= 10] = 0  # Make binary with low threshold
    RGBnp[RGBnp > 10] = 1
    BinaryNP = RGBnp[:, :, 0]  # [0,0,0]->[0]

    binary_image = BinaryNP
    pixel_sum = binary_image  # .sum(-1)  # sum over color (last) axis
    nonzero_pixels = (pixel_sum > 0).astype(bool)
    W = morphology.remove_small_objects(nonzero_pixels, 20, connectivity=2)
    W = W.astype(float)

    im = input_image_obj
    input_image_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresholded_image = cv2.threshold(input_image_gray, 127, 255, 0)

    if Side == "Left":
        ROIAX = thresholded_image[
            int(Left_dimensions[2]) : int(Left_dimensions[3]),
            int(Left_dimensions[0]) : int(Left_dimensions[1]),
        ]  # Right ROI
        TGT = 48
    elif Side == "Right":
        TGT = 2
        ROIAX = thresholded_image[
            int(Right_dimensions[2]) : int(Right_dimensions[3]),
            int(Right_dimensions[0]) : int(Right_dimensions[1]),
        ]  # Left ROI

    ROI2 = np.zeros(np.shape(ROIAX))
    ROI3 = np.zeros(np.shape(ROIAX))
    # plt.imshow(ROI)
    # plt.show()

    contours, hierarchy = cv2.findContours(
        ROIAX, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(ROI2, contours, -1, [255], 1)
    Cs = list(contours)  # list the contour coordinates as array
    if Side == "Right":
        for Column in range(
            0, int(Right_dimensions[1]) - int(Right_dimensions[0])
        ):  # Move across and find the line with most contours in.
            count = 0
            for Row in range(0, (int(Right_dimensions[3]) - int(Right_dimensions[2]))):
                if ROI2[Row, Column] == 255:
                    count = count + 1

            if count > (15):
                ROI2[:, Column] = 0

    ROI2 = ROI2.astype(np.uint8)
    contours, hierarchy = cv2.findContours(
        ROI2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    cv2.drawContours(ROI2, contours, -1, [255], 1)
    Cs = list(contours)  # list the contour coordinates as array
    CsX = []  # X coord of the center of each contour
    CsY = []  # X coord of the center of each contour
    for C in Cs:
        # compute the center of the contour
        cx = 0
        cy = 0
        for rgb_values in C:
            cx += rgb_values[0][0]
            cy += rgb_values[0][1]
        CsX.append(int(cx / len(C)))
        CsY.append(int(cy / len(C)))

    I = len(Cs)  # number of contours
    lengths = []  # Initialise variable to fill with contour lengths

    for i in range(0, I):
        lengths.append(Cs[i].size)  # fill with contour lengths

    lengths = np.array(lengths)  # make an array
    ids = np.where(lengths > 0)  # indexes of the lengths greater than 0
    ids = ids[0]  # because ids looks like (array([...])) and we want array([...])
    ids = ids[
        ::-1
    ]  # reverse order so indexes run fron high to low, this is needed for the next loop
    onY, BCs, Xs, EndPoints, CenPoints = (
        [],
        [],
        [],
        [],
        [],
    )  # Initialise some variables

    all = []
    for TGT in range(0, int(Right_dimensions[1]) - int(Right_dimensions[0])):
        count = 0
        Ctest = 0
        for id in ids:
            Ctest = np.reshape(Cs[id], (-1, 2))
            x_values = [i[0] for i in Ctest]
            if TGT in x_values:
                count = count + 1

        all.append(count)

    peaks, vals = signal.find_peaks(all, height=3)  # Miss last 20 pixels as
    if Side == "Left":
        maxID = np.argmax(peaks)
    elif Side == "Right":
        peak_wid = signal.peak_widths(all, peaks)
        maxID = np.argmax(vals["peak_heights"] * peak_wid[0])

    TGT = peaks[maxID]
    # TGT = all.index(max(all)) # The target is the X coord that most object lie on.

    for id in ids:
        Ctest = np.reshape(Cs[id], (-1, 2))
        x_values = [i[0] for i in Ctest]
        if (
            TGT in x_values
        ):  # Looks if any contour coords are on a line (2,:), this is close to ROI bounds but not in contact.
            tempXs, tempYs = (
                [],
                [],
            )  # Clear/Initialise temporary X Y coordinate stores
            # cv2.drawContours(ROI3, Cs[id], -1,[255], 1)
            onY.append(id)  # record the contour index that meets criteria
            BCs.append(Cs[id])  # ?
            for l in range(0, len(Cs[id])):
                Xs.append(Cs[id][l][0][0])  # Needed?
                tempXs.append(Cs[id][l][0][0])  # Store X coords
                tempYs.append(Cs[id][l][0][1])  # Store Y coords

            MAXX = max(tempXs)  # Max X from this contour
            MINX = min(tempXs)  # Min X from this contour
            MAXY = max(tempYs)  # Max Y from this contour
            MINY = min(tempYs)  # Min Y from this contour
            if Side == "Left":
                index = tempXs.index(
                    MINX
                )  # Index of the max X - this is the "end point" of Side == "Right"
                EndPoints.append(
                    tempXs[index]
                )  # Save end point (Might be redundant with new method?)
            elif Side == "Right":
                index = tempXs.index(
                    MAXX
                )  # Index of the max X - this is the "end point" of Side == "Right"
                EndPoints.append(
                    tempXs[index]
                )  # Save end point (Might be redundant with new method?)
            CenPoints.append(
                [int((MAXX + MINX) / 2), int((MAXY + MINY) / 2)]
            )  # Calc center point as (0.5*(MaxX+MinX),0.5*(MaxY+MinY))

    def reject_outliers(data, m=8.0):
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d / (mdev if mdev else 1.0)
        outdata = []
        badIDS = []
        for i in range(0, len(data)):
            if s[i] < m:
                outdata.append(data[i])
            else:
                badIDS.append(i)

        return outdata, badIDS

    if Side == "Right":
        TYLshift = max(
            EndPoints
        )  # The shift reduces the ROIAX to avoid intaining the ticks, as these can be confused as '-' symbols
    elif Side == "Left":
        EndPoints, badIDS = reject_outliers(EndPoints)
        try:
            # cv2.drawContours(ROI3, Cs[badIDS[0]], -1,[255], 1)
            CenPoints.pop(badIDS[0])
            Cs.pop(badIDS[0])
            onY.pop(badIDS[0])
            BCs.pop(badIDS[0])
        except Exception:
            pass

        TYLshift = min(
            EndPoints
        )  # The shift reduces the ROIAX to avoid intaining the ticks, as these can be confused as '-' symbols
    Cs = tuple(Cs)  # Change to tuple?

    if Side == "Left":
        ROIAX = thresholded_image[
            int(Left_dimensions[2]) : int(Left_dimensions[3]),
            int(Left_dimensions[0]) : int(Left_dimensions[0] + TYLshift),
        ]  # Right ROI
    elif Side == "Right":
        ROIAX = thresholded_image[
            int(Right_dimensions[2]) : int(Right_dimensions[3]),
            int(Right_dimensions[0] + TYLshift) : int(Right_dimensions[1]),
        ]  # Left ROI

    return (
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
    )


def Search_for_labels(
    Cs,
    ROIAX,
    CenPoints,
    onY,
    BCs,
    TYLshift,
    Side,
    Left_dimensions,
    Right_dimensions,
    input_image_obj,
    ROI2,
    ROI3,
):
    for thresh_value in np.arange(100, 190, 5): # Threshold to optimise the resulting text extraction.
        image = input_image_obj
        input_image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresholded_image = cv2.threshold(input_image_gray, thresh_value, 255, 0)
        if Side == "Left":
            ROIAX = thresholded_image[
                int(Left_dimensions[2]) : int(Left_dimensions[3]),
                int(Left_dimensions[0]) : int(Left_dimensions[0] + TYLshift),
            ]  # Right ROI
        elif Side == "Right":
            ROIAX = thresholded_image[
                int(Right_dimensions[2]) : int(Right_dimensions[3]),
                int(Right_dimensions[0] + TYLshift) : int(Right_dimensions[1]),
            ]  # Left ROI

        extracted_text_data = pytesseract.image_to_data(
            ROIAX,
            output_type=Output.DICT,
            config="--psm 11 -c tessedit_char_whitelist=-0123456789",
        )
        number = []
        for i in range(len(extracted_text_data["text"])):
            if extracted_text_data["text"][i] != "":
                number.append(extracted_text_data["text"][i])

        retry = 0
        for num in number:
            try:
                if (float(num) / 5).is_integer() == 0:
                    retry += 1
            except Exception:
                pass

        if retry == 0:
            break

    # d = pytesseract.image_to_data(
    #     ROIAX,
    #     output_type=Output.DICT,
    #     config="--psm 11 -c tessedit_char_whitelist=-0123456789",
    # )
    n_boxes = len(extracted_text_data["level"])
    CenBox = []  # Initialise variable to populate with box center coords
    for i in range(1, n_boxes):  # dont start from 0 as the first index is redundant
        if Side == "Left":
            (x, y, wi, h) = (
                extracted_text_data["left"][i],
                extracted_text_data["top"][i],
                extracted_text_data["width"][i],
                extracted_text_data["height"][i],
            )  # define (Xleft, Ytop, width, height) of each object from the dictionary
        elif Side == "Right":
            (x, y, wi, h) = (
                extracted_text_data["left"][i] + TYLshift,
                extracted_text_data["top"][i],
                extracted_text_data["width"][i],
                extracted_text_data["height"][i],
            )  # define (Xleft, Ytop, width, height) of each object from the dictionary

        o = i / 4  # we get 4 repeats for each real box, so this reduces that to 1.
        if o.is_integer():
            if Side == "Left":
                CenBox.append(
                    [
                        (extracted_text_data["left"][i] + (extracted_text_data["width"][i] / 2)),
                        (extracted_text_data["top"][i] + (extracted_text_data["height"][i] / 2)),
                    ]
                )  # calculate the center point of each bounding box
            elif Side == "Right":
                CenBox.append(
                    [
                        (extracted_text_data["left"][i] + TYLshift + (extracted_text_data["width"][i] / 2)),
                        (extracted_text_data["top"][i] + (extracted_text_data["height"][i] / 2)),
                    ]
                )  # calculate the center point of each bounding box
        cv2.rectangle(
            ROI3, (x, y), (x + wi, y + h), (255), 2
        )  # Draw the rectangles on the ROI

    for i in range(0, len(CenBox)):
        dists = cdist([CenBox[i]], CenBox)
        dists[0][i] = dists.max()
        if dists.min() < 20: # detect if characters are too close - but what if they are?
            logger.warning("Characters too close")

    try:
        # Find the nearest tick for each number
        dist = []  # Initlise distance variable
        Mindex = np.zeros(len(CenBox))  # Initlialise Min index variable
        for txt in range(0, len(CenBox)):  # For all axes label text boxes
            for tck in range(0, len(CenPoints)):  # For all axes ticks
                dist.append(
                    math.sqrt(
                        (CenBox[txt][0] - CenPoints[tck][0]) ** 2 + (CenBox[txt][1] - CenPoints[tck][1]) ** 2
                    )
                )  # Distance between current text box and all ticks
            MIN = min(dist)  # Find the shortest distance to a tick
            Mindex[txt] = dist.index(
                MIN
            )  # Identify the index of the tick at the nearest distance and store.
            dist = []  # Clear dist variable

        positions = []  # Initialise position variable
        for id in Mindex:  # for each index in the Min index store
            # Store the closest centerpoints as found in the previous loop
            #  add ROI components to make positions relative to overall image.
            if Side == "Left":  # Adjust for the left side
                positions.append(
                    [
                        CenPoints[int(id)][0] + int(Left_dimensions[0]),
                        CenPoints[int(id)][1] + int(Left_dimensions[2]),
                    ]
                )
            elif Side == "Right":  # Or adjust for the right side
                positions.append(
                    [
                        CenPoints[int(id)][0] + int(Right_dimensions[0]),
                        CenPoints[int(id)][1] + int(Right_dimensions[2]),
                    ]
                )

        Mindex = Mindex[::-1]  # Reverse order so runs from high to low

        # for id in Mindex: # Common sense check - are all ticks evenly space?
        #     Failed_Indexes.append(BCs[int(id)]) # Order ticks lowest to highest?

        Final_TickIDS = []  # Init variable to store final tick indexs
        Final_CenPoints = []
        IDSL = []
        Mindex = sorted(Mindex, reverse=True)
        for id in Mindex:
            Final_TickIDS.append(BCs[int(id)])  # Order ticks lowest to highest?
            IDSL.append(int(id))
            Final_CenPoints.append(CenPoints[int(id)])

    except Exception:  # if this step fails, a backup is to assume center of text box is the tick
        extracted_text_data = pytesseract.image_to_data(
            ROIAX,
            output_type=Output.DICT,
            config="--psm 11 -c tessedit_char_whitelist=-0123456789",
        )
        n_boxes = len(extracted_text_data["level"])
        CenBox = []  # Initialise variable to populate with box center coords
        for i in range(1, n_boxes):  # dont start from 0 as the first index is redundant
            if Side == "Left":
                (x, y, wi, h) = (
                    extracted_text_data["left"][i],
                    extracted_text_data["top"][i],
                    extracted_text_data["width"][i],
                    extracted_text_data["height"][i],
                )  # define (Xleft, Ytop, width, height) of each object from the dictionary
            elif Side == "Right":
                (x, y, wi, h) = (
                    extracted_text_data["left"][i] + TYLshift,
                    extracted_text_data["top"][i],
                    extracted_text_data["width"][i],
                    extracted_text_data["height"][i],
                )  # define (Xleft, Ytop, width, height) of each object from the dictionary

            o = i / 4  # we get 4 repeats for each real box, so this reduces that to 1.
            if o.is_integer():
                if Side == "Left":
                    CenBox.append(
                        [
                            (extracted_text_data["left"][i] + (extracted_text_data["width"][i] / 2)),
                            (extracted_text_data["top"][i] + (extracted_text_data["height"][i] / 2)),
                        ]
                    )  # calculate the center point of each bounding box
                elif Side == "Right":
                    CenBox.append(
                        [
                            (extracted_text_data["left"][i] + TYLshift + (extracted_text_data["width"][i] / 2)),
                            (extracted_text_data["top"][i] + (extracted_text_data["height"][i] / 2)),
                        ]
                    )  # calculate the center point of each bounding box
            cv2.rectangle(
                ROI3, (x, y), (x + wi, y + h), (255), 2
            )  # Draw the rectangles on the ROI

        for i in range(0, len(CenBox)):
            dists = cdist([CenBox[i]], CenBox)
            dists[0][i] = dists.max()
            if dists.min() < 20:
                logger.warning("Characters too close")

        Final_CenPoints = CenBox

    # Failed_Indexes = []

    dists = cdist(Final_CenPoints, Final_CenPoints)

    def index_list(dists):
        lst = list(dists)
        length = len(lst)
        dist = []
        for i in lst:
            if lst.index(0) > lst.index(i):
                diff = lst.index(0) - lst.index(i)
                dist.append(diff)
            elif lst.index(0) < lst.index(i):
                diff = abs(lst.index(i)) - abs(lst.index(0))
                dist.append(diff)
            elif lst.index(0) == lst.index(i):
                dist.append(0)

        dist_divided = []
        for i in range(0, length):
            dist_divided.append(lst[i] / dist[i])

        return dist_divided

    # ordered_indexs1 = index_list(dists[0])
    # ordered_indexs2 = index_list(dists[1])
    # ordered_indexs3 = index_list(dists[2])
    # ordered_indexs4 = index_list(dists[3])
    # ordered_indexs5 = index_list(dists[4])
    # ordered_indexs6 = index_list(dists[5])

    cv2.drawContours(ROI3, Final_TickIDS, -1, [255], 1)

    number = []
    for i in range(len(extracted_text_data["text"])):
        if extracted_text_data["text"][i] != "":
            number.append(extracted_text_data["text"][i])

    empty_to_fill = np.zeros((image.shape[0], image.shape[1]))

    if Side == "Left":
        empty_to_fill[
            int(Left_dimensions[2]) : int(Left_dimensions[3]),
            int(Left_dimensions[0]) : int(Left_dimensions[1]),
        ] = ROI3  # Right ROI
    elif Side == "Right":
        empty_to_fill[
            int(Right_dimensions[2]) : int(Right_dimensions[3]),
            int(Right_dimensions[0]) : int(Right_dimensions[1]),
        ] = ROI3  # Left ROI

    return ROIAX, number, positions, empty_to_fill


def Plot_Digitized_data(Rticks, Rlocs, Lticks, Llocs, top_curve_coords):
    # Function to digitize the segmetned data:

    Rticks = list(map(int, Rticks))
    XmaxtickR = max(Rticks)
    XmaxidR = Rticks.index(XmaxtickR)
    XmaxR = Rlocs[XmaxidR][0]
    YmaxR = Rlocs[XmaxidR][1]
    XMintickR = min(Rticks)
    XMinidR = Rticks.index(XMintickR)
    XminR = Rlocs[XMinidR][0]
    #
    Lticks = list(map(int, Lticks))
    XmaxtickL = max(Lticks)
    XmaxidL = Lticks.index(XmaxtickL)
    XmaxL = Llocs[XmaxidL][0]
    XMintickL = min(Lticks)
    XminidL = Lticks.index(XMintickL)
    XminL = Llocs[XminidL][0]
    YminL = Llocs[XminidL][1]

    # Yplots = [Llocs[XmaxidL][0], Llocs[XminidL][0], Rlocs[XmaxidR][0]]
    # Xplots = [Llocs[XmaxidL][1], Llocs[XminidL][1], Rlocs[XmaxidR][1]]

    Xmin = 0
    Xmax = 1
    Ymin = XMintickL
    Ymax = XmaxtickL

    b = top_curve_coords
    b = sorted(b, key=lambda k: [k[1], k[0]])

    b = [B.tolist() for B in b]
    b = [x[::-1] for x in b]

    b = pd.DataFrame(b).groupby(0, as_index=False)[1].mean().values.tolist()
    b = [x[::-1] for x in b]

    X = [XminL, XmaxR]  
    Y = [YminL, YmaxR] 

    for i in range(0, len(b)):
        if b[i][1] >= XminL + 20 and b[i][1] <= XmaxR - 20:
            X.append(b[i][1])
            Y.append(b[i][0])

    origin = [X[0], Y[0]]
    topRight = [X[1], Y[1]]
    XminScale = origin[0]
    XmaxScale = topRight[0]
    YminScale = origin[1]
    YmaxScale = topRight[1]

    Ynought = [(0 - YminScale) / (YmaxScale - YminScale) * (Ymax - Ymin) + Ymin]

    X = X[2:-1]
    Y = Y[2:-1]

    Xplot = [
        (i - XminScale) / (XmaxScale - XminScale) * (Xmax - Xmin) + Xmin for i in X
    ]
    Yplot = [
        (i - YminScale) / (YmaxScale - YminScale) * (Ymax - Ymin) + Ymin for i in Y
    ]

    # Inverts the waveform if need be
    if np.mean(Yplot) < 0:
        Yplot = [ y * (-1) for y in Yplot]

    plt.figure(2)
    plt.plot(Xplot, Yplot, "-")
    plt.xlabel("Arbitrary time scale")
    plt.ylabel("Flowrate (cm/s)")
    return Xplot, Yplot, Ynought


def Plot_correction(Xplot, Yplot, df):
    """
    Use data from text extraction to correct the plot
    """

    try:
        # import Jinja2
        y = np.array(Yplot)
        x = np.array(Xplot)
        # DF = df
        df.insert(loc=3, column="Digitized Value", value="")
        peaks, _ = find_peaks(y)  # PSV
        troughs, _ = find_peaks(-y)  # EDV
        # filter out any anomylous signals:
        trough_widths, _, _, _ = peak_widths(-y, troughs)
        mean_widths_troughs = statistics.mean(trough_widths)
        # filter out anomalous peaks
        valid_troughs = []
        for i in range(len(troughs)):
            if trough_widths[i] > (mean_widths_troughs / 2):
                valid_troughs.append(troughs[i])
        troughs = valid_troughs

        widths_of_peaks, _, _, _ = peak_widths(y, peaks)
        mean_widths_peaks = statistics.mean(widths_of_peaks)
        valid_peaks = []
        for i in range(len(peaks)):
            if widths_of_peaks[i] > (mean_widths_peaks / 2) and y[peaks[i]] > (
                statistics.mean(y[peaks]) * 0.8
            ):
                valid_peaks.append(peaks[i])
        peaks = valid_peaks

        # Peak systolic
        PS = statistics.mean(y[peaks])
        # End diastolic
        ED = statistics.mean(y[troughs])
        # Find S/D
        SoverD = statistics.mean(y[peaks]) / statistics.mean(y[troughs])
        # Find RI
        RI = (
            statistics.mean(y[peaks]) - statistics.mean(y[troughs])
        ) / statistics.mean(y[peaks])
        # Find TAmax
        TAmax = (statistics.mean(y[peaks]) + (2 * statistics.mean(y[troughs]))) / 3
        # Find PI
        PI = (
            statistics.mean(y[peaks]) - statistics.mean(y[troughs])
        ) / statistics.mean(y)

        words = ["PS", "ED", "S/D", "RI", "TA"]
        values = [PS, ED, SoverD, RI, TAmax]
        values = [round(elem, 2) for elem in values]
        # Loop through each word and value
        for i in range(len(words)):
            word = words[i]
            value = values[i]
            try:
                # Find rows where word is in "Text" column, and set corresponding "Value" to the value
                df.loc[df["Word"].str.contains(word), "Digitized Value"] = value

                # get_colour(df.loc[df["Word"].str.contains(word),'Value'].values[0],value)
            except Exception:
                continue

        # Period of the signal in arbitrary scale
        arbitrary_period = (x[peaks[-1]] - x[peaks[1]]) / (len(peaks) - 2)
    except Exception:
        traceback.print_exc()  # prints the error message and traceback

    try:
        # Period of the signal in real time scale from text extraction
        real_period = 1 / (
            df.loc[df["Word"].str.contains("HR"), "Value"].values[0] / 60
        )
        # Calculate a scaling factor
        scale_factor = real_period / arbitrary_period
        x_time = x * scale_factor
        plt.close(2)
        plt.figure(2)
        plt.plot(x_time, y)
        plt.plot(x_time[peaks], y[peaks], "x")
        plt.plot(x_time[troughs], y[troughs], "x")
        # plt.plot(x_time[troughs[1:-1]],y[troughs[1:-1]],"x")
        plt.xlim((min(x_time), max(x_time)))
        plt.ylim((0, max(y) + 10))
        plt.xlabel("Time (s)")
        plt.ylabel("Flowrate (cm/s)")
    except Exception:
        traceback.print_exc()

    return df


def Annotate(
    input_image_obj,
    refined_segmentation_mask,
    Left_dimensions,
    Right_dimensions,
    Waveform_dimensions,
    Left_axis,
    Right_axis,
):
    # Function to annotate original image with the components computed in each of the previous functions.

    Xmin, Xmax, Ymin, Ymax = Waveform_dimensions
    Xmin_R, Xmax_R, Ymin_R, Ymax_R = Right_dimensions
    Xmin_L, Xmax_L, Ymin_L, Ymax_L = Left_dimensions

    rR = [Xmin_R, Xmax_R, Xmax_R, Xmin_R, Xmin_R]
    cR = [Ymax_R, Ymax_R, Ymin_R, Ymin_R, Ymax_R]
    rrR, ccR = polygon_perimeter(cR, rR, refined_segmentation_mask.shape)

    rL = [Xmin_L, Xmax_L, Xmax_L, Xmin_L, Xmin_L]
    cL = [Ymax_L, Ymax_L, Ymin_L, Ymin_L, Ymax_L]
    rrL, ccL = polygon_perimeter(cL, rL, refined_segmentation_mask.shape)

    r = [Xmin, Xmax, Xmax, Xmin, Xmin]
    c = [Ymax, Ymax, Ymin, Ymin, Ymax]
    rr, cc = polygon_perimeter(c, r, refined_segmentation_mask.shape)

    refined_segmentation_mask[rr, cc] = 2  # set color white
    refined_segmentation_mask[rrL, ccL] = 2
    refined_segmentation_mask[rrR, ccR] = 2

    img_RGB = input_image_obj  # .convert('RGB')
    pixel_data = img_RGB.load()
    # gry = img_RGB.convert("L")  # returns grayscale version.

    for y in range(img_RGB.size[1]):
        for x in range(img_RGB.size[0]):
            # r = pixel_data[x, y][0]
            # g = pixel_data[x, y][1]
            # b = pixel_data[x, y][2]
            # rgb_values = [r, g, b]
            # min_rgb = min(rgb_values)
            # max_rgb = max(rgb_values)
            # rgb_range = max_rgb - min_rgb

            if refined_segmentation_mask[y, x] == 1:
                pixel_data[x, y] = (
                    255,
                    pixel_data[x, y][1],
                    pixel_data[x, y][2],
                    250,
                )  # Segmented waveform as Red
            elif refined_segmentation_mask[y, x] == 2:
                pixel_data[x, y] = (1, 255, 1, 255)  # Set ROIs to blue
            elif Left_axis[y, x] == 255:
                pixel_data[x, y] = (255, 0, 255, 255)  # Set ROIs to blue
            elif Right_axis[y, x] == 255:
                pixel_data[x, y] = (255, 0, 255, 255)  # Set ROIs to blue
            else:
                pixel_data[x, y] = pixel_data[x, y]
    return img_RGB


@lru_cache(maxsize=16384)
def _in_cylinder(coord, bot, top, center, const):
    coord = np.array(coord).reshape(1, 3)
    above_bot = np.dot(coord - bot, center) >= 0
    below_top = np.dot(coord - top, center) <= 0
    within_rad = np.linalg.norm(np.cross(coord - bot, center.T)) <= const
    return (above_bot and below_top and within_rad).item()


def Colour_extract(input_image_obj, TargetRGB, cyl_length, cyl_radius):
    """Extract target colours from image.

    Args:
        input_image_obj (PIL Image) : PIL Image object.
        TargetRGB (list) : triplet of target Red Green Blue colour [Red,Green,Blue].
        cyl_length (int) : length of cylinder.
        cyl_radius (int) : radius of cylinder.

    Returns:
        COL (JpegImageFile) : PIL JpegImageFile of the filtered image highlighting selected text.
    """

    # Finds the minimum and maximum magnitudes for the colour vector
    img = np.array(input_image_obj)[:, :, :3].astype(float)
    rgb_vec = np.array(TargetRGB)
    mag = np.sqrt(np.sum(rgb_vec ** 2))
    img_dot = np.zeros_like(img[:, :, 0])

    for channel, val in enumerate(rgb_vec):
        img_dot += img[:, :, channel] * val
    img_dot /= mag

    min_mag = mag - cyl_length / 2
    max_mag = mag + cyl_length / 2

    # Finds the distance of the colours to the cylinder axis
    img_cross = np.zeros_like(img_dot)
    for channel, val in enumerate(rgb_vec):
        c1 = channel + 1 if channel + 1 < len(rgb_vec) else channel + 1 - len(rgb_vec)
        c2 = channel + 2 if channel + 2 < len(rgb_vec) else channel + 2 - len(rgb_vec)
        img_cross += (rgb_vec[c1] * img[:, :, c2] - rgb_vec[c2] * img[:, :, c1]) ** 2
    distance_to_axis = np.sqrt(img_cross) / mag
    
    mask = np.logical_and(
        np.logical_and(img_dot <= max_mag, img_dot >= min_mag),
        distance_to_axis <= cyl_radius,
    ).astype(np.uint8) * 255

    return Image.fromarray(mask)


def Text_from_greyscale(input_image_obj, COL):
    """
    Function for extracting text from the yellow filtered image.

    Args:
        input_image_filename (str) : Name of file within current directory, or path to a file.
        COL (JpegImageFile) : PIL JpegImageFile of the filtered image highlighting yellow text.
    Returns:
        Fail (int) : Checks if the function has failed (1), or passed (0).
        df (DataFrame) : Dataframe with columns 'Line', 'Word', 'Value', 'Unit'.
        populated with data extracted from the image with tesseract.

    """

    PIX = COL.load()
    img = input_image_obj
    for y in range(
        int(COL.size[1] * 0.45), COL.size[1]
    ):  # Exclude bottom 3rd of image - these are fails
        for x in range(COL.size[0]):
            PIX[x, y] = (0, 0, 0)


    pixels = np.array(COL)
    data = pytesseract.image_to_data(
        pixels, output_type=Output.DICT, lang="eng", config="--psm 3 "
    )
    if (
        len(data["text"]) < 30
    ):  # This is rough, if more than 30 objects found then highly likley it is a waveform scan.
        Fail = 1  # logical, keep scan or not
    else:
        Fail = 0

    # Loop through each word and draw a box around it
    y_center = np.zeros(len(data["text"])) # Variable to store the y-center of each bounding box of text detected.
    for i in range(len(data["text"])):
        if data["text"][i] != '':
            x = data["left"][i]
            y = data["top"][i]
            w = data["width"][i]
            h = data["height"][i]
            if int(data["conf"][i]) > 1:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                y_center[i] = y + (h/2)
            else:
                y_center[i] = 0  
        else:
            y_center[i] = 0
        

    def group_similar_numbers(array, tolerance, words):
        # This function groups indexes of words with similar y-coordinate center
        # Assuming that this is an indicator of words being part of the same line
        # Prepare the data for clustering
        data = np.array(array).reshape(-1, 1)

        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=tolerance, min_samples=2)
        labels = dbscan.fit_predict(data)

        # Group the indices based on the cluster labels
        groups = {}
        for i, label in enumerate(labels):
            if label in groups:
                groups[label].append(i)
            else:
                groups[label] = [i]

        # Add the words together for each group to make each line of text
        grouped_words = []
        for group in groups.values():
            group_words = [words[idx] for idx in group]
            grouped_words.append(' '.join(group_words))

        return grouped_words

    tolerance = 3  # Adjust the tolerance value - the max difference between y-coords considered on the same line
    grouped_words = group_similar_numbers(y_center, tolerance, data["text"])

    # Display image
    plt.imshow(img)

    # Analyze the OCR output
    target_words = [
        "Lt Ut-PS",
        "Lt Ut-ED",
        "Lt Ut-S/D",
        "Lt Ut-PI",
        "Lt Ut-RI",
        "Lt Ut-MD",
        "Lt Ut-TAmax",
        "Lt Ut-HR",
        "Rt Ut-PS",
        "Rt Ut-ED",
        "Rt Ut-S/D",
        "Rt Ut-PI",
        "Rt Ut-RI",
        "Rt Ut-MD",
        "Rt Ut-TAmax",
        "Rt Ut-HR",
        "Umb-PS",
        "Umb-ED",
        "Umb-S/D",
        "Umb-PI",
        "Umb-RI",
        "Umb-MD",
        "Umb-TAmax",
        "Umb-HR",
    ]

    # Split text into lines
    lines = grouped_words #text.split("\n")

    # Initialize DataFrame
    df = pd.DataFrame(columns=["Line", "Word", "Value", "Unit"])

    # Loop over lines
    for i, line in enumerate(lines):
        # Loop over target words
        for word in target_words:
            # Look for word in line
            if word in line:
                # Extract value and unit
                match = re.search(r"(\-?\d+\.\d+|\-?\d+)\s*([^\d\s]+)?$", line)
                if match:
                    value = float(match.group(1))
                    unit = match.group(2) if match.group(2) else ""
                    df = df._append(
                        {"Line": i + 1, "Word": word, "Value": value, "Unit": unit},
                        ignore_index=True,
                    )
                else:
                    logger.warning("couldn't find numeric data for line.")
                    df = df._append(
                        {"Line": i + 1, "Word": word, "Value": 0, "Unit": 0},
                        ignore_index=True,
                    )

    return Fail, df

def Scan_type_test(input_image_filename):
    """
    Function for yellow filtering an image and searching for a list of target words indicative of a doppler ultrasound scan

    Args:
        input_image_filename (str) : Name of file within current directory, or path to a file.

    Returns:
        Fail (int) : Idicates if the file is a fail (1) - doesn't meet criteria for a doppler ultrasound, or pass (0) - does meet criteria. 

    """

    img = cv2.imread(input_image_filename)  # Input image file
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Convert to HSV
    lower_yellow = np.array([1, 100, 100], dtype=np.uint8)  # Lower yellow bound
    upper_yellow = np.array([200, 255, 255], dtype=np.uint8)  # Upper yellow bound
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)  # Threshold HSV between bounds
    yellow_text = cv2.bitwise_and(gray, gray, mask=mask)

    yellow_text[int(img.shape[1] * 0.45): img.shape[1], :] = 0  # Exclude bottom 3rd of image - target scans have no text of interest here.
    pixels = np.array(yellow_text)
    data = pytesseract.image_to_data(
        pixels, output_type=Output.DICT, lang="eng", config="--psm 3 "
    )

    # Loop through each word and draw a box around it
    for i in range(len(data["text"])):
        x = data["left"][i]
        y = data["top"][i]
        segmentation_mask = data["width"][i]
        h = data["height"][i]
        if int(data["conf"][i]) > 1:
            cv2.rectangle(img, (x, y), (x + segmentation_mask, y + h), (0, 0, 255), 2)

    # Display image
    # cv2.imshow('img', img)

    # Perform OCR on the preprocessed image
    custom_config = r"--oem 3 --psm 3"
    text = pytesseract.image_to_string(pixels, lang="eng", config=custom_config)

    # Analyze the OCR output
    lines = text.splitlines()
    target_words = [
        "HR",
        "TAmax",
        "Lt Ut-PS",
        "Lt Ut-ED",
        "Lt Ut-S/D",
        "Lt Ut-PI",
        "Lt Ut-RI",
        "Lt Ut-MD",
        "Lt UT-TAmax",
        "Lt Ut-HR",
        "Rt Ut-PS",
        "Rt Ut-ED",
        "Rt Ut-S/D",
        "Rt Ut-PI",
        "Rt Ut-RI",
        "Rt Ut-MD",
        "Rt UT-TAmax",
        "Rt Ut-HR",
        "Umb-PS",
        "Umb-ED",
        "Umb-S/D",
        "Umb-PI",
        "Umb-RI",
        "Umb-MD",
        "Umb-TAmax",
        "Umb-HR",
    ]  # Target words to search for - there may be more to add to this.

    # Split text into lines
    lines = text.split("\n")

    # Initialize DataFrame
    df = pd.DataFrame(columns=["Line", "Word", "Value", "Unit"])

    Fail = 1  # initialise fail variable
    for target in target_words:
        for word in lines:
            if target in word:
                Fail = 0 # If any of the words are found, then no fail.

    return Fail, df  # Return the fail variable and dataframe contraining extracted text.

