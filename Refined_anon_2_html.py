import matplotlib.pyplot as plt
import pytesseract
from PIL import Image
import traceback
import os
import pickle

# Import sementation module
import General_functions

pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"

# If you have ran the Organise_files.py script, load the output here:
# # Open the file containing file names of target scans - created by running Organise_files.py
# with open("patient_paths_test.pkl", "rb") as f:
#     text_file = pickle.load(f)

# # Define which patients to process (between key1 and key2)
# key1 = "0000"
# #key2 = "0039"
# # Get a list of all the keys in the dictionary
# keys = list(text_file.keys())

# try:
#     # Get the index of the first key
#     idx1 = keys.index(key1)
#     # Get the index of the second key
#     idx2 = keys.index(key2)
#     # Get the sublist of keys between the two keys
#     subkeys = keys[idx1:idx2 + 1]
# except Exception:  # If the specified keys dont exist, default to all keys.
#     subkeys = keys

# filenames = []
# # Iterate through the sublist of keys
# for key in subkeys:
#     # Access the value corresponding to the key
#     filenames = filenames + text_file[key]
#     # print(filenames)

# If you want to run a single file:
filenames = ['C:/Users/dalek/OneDrive/Ultasound_segmentation/Lt_test_image.png'] #"Path/To/Image.jpeg"

output_path = "E:/us-data-processed/"  # where you want to save data
# Inititalise some variables
Text_data = []  # text data extracted from image
Annotated_scans = []
Digitized_scans = []

for input_image_filename in filenames:  # Iterare through all file names and populate excel file
    # input_image_filename = "To/test/one/file.jpg"
    image_name = os.path.basename(input_image_filename)
    print(input_image_filename)

    try:  # Try text extraction
        colRGBA = Image.open(input_image_filename)  # These images are in RGBA form
        col = colRGBA.convert("RGB")  # We need RGB, so convert here.
        pix = (
            col.load()
        )  # Loads a pixel access object, where pixel values can be edited

        # from General_functions import Colour_extract, Text_from_greyscale
        COL = General_functions.Colour_extract(input_image_filename, [255, 255, 100], 80, 80)
        print("Done Colour extract")

        Fail, df = General_functions.Text_from_greyscale(input_image_filename, COL)
    except Exception:  # If text extraction fails
        traceback.print_exc()  # prints the error message and traceback
        print("Failed Text extraction")
        Text_data.append(None)
        Fail = 0
        pass

    try:  # Try initial segmentation
        segmentation_mask, Xmin, Xmax, Ymin, Ymax = General_functions.Initial_segmentation(
            input_image_filename=input_image_filename
        )
    except Exception:  # If initial segmentation fails
        print("Failed Initial segmentation")
        Fail = Fail + 1
        pass

    try:  # Define end ROIs
        Left_dimensions, Right_dimensions = General_functions.Define_end_ROIs(
            segmentation_mask, Xmin, Xmax, Ymin, Ymax
        )
        Waveform_dimensions = [Xmin, Xmax, Ymin, Ymax]
    except Exception:  # If defing ROI fails
        print("Failed Defining ROI")
        Fail = Fail + 1
        pass

    try:  # Refine segmentation
        refined_segmentation_mask, top_curve_mask = General_functions.Segment_refinement(
            input_image_filename, Xmin, Xmax, Ymin, Ymax
        )
    except Exception:
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
            input_image_filename, "Left", Left_dimensions, Right_dimensions
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
            input_image_filename, "Right", Left_dimensions, Right_dimensions
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
        Annotated_path = output_path + image_name.partition(".")[0] + "_Annotated.png"

        #  Create annotated figure for analysis
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
            print("Failed correction")
            continue
        Digitized_path = output_path + image_name.partition(".")[0] + "_Digitized.png"

        # Open figure initialised in Plot_correction function
        plt.figure(2)
        plt.savefig(Digitized_path, dpi=900, bbox_inches="tight", pad_inches=0)
        Digitized_scans.append(Digitized_path)
    except Exception:
        print("Failed Digitization")
        try:
            Text_data.append(df)
        except Exception:
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

# Save the processed data 
with open("solo_test.pickle", "wb") as f:
    pickle.dump([filenames, Digitized_scans, Annotated_scans, Text_data], f)
i = 0
