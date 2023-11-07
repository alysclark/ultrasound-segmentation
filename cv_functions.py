""" A set of functions that use computer vision to extract metadata"""
#! /usr/bin/env python

# Python imports
import os
from PIL import Image

# Module imports
import numpy as np
import pytesseract
import skimage.filters
import toml
from loguru import logger
import usseg


# Loads in ultrasound templates from TOML file
def load_us_templates(toml_file, template_section="templates"):
    """Loads the institute ultrasound templates from a TOML file.

    Args:
        toml_file (str) : A string to the TOML file.
        template_section (str, optional) : The heading of the templates section.
            Defaults to "templates".

    Returns:
        template_list (tuple) : A list of dictionaries containing the ultrasound templates.
    """

    # Loads in the TOML file
    toml_out = toml.load(toml_file)
    templates = toml_out[template_section]

    # Iterates over templates, storing them in a list
    template_list = []
    for template in templates:
        template_list.append(templates[template])

    return template_list


def load_default_config_file_dict():
    """Loads the default configuration file dictionary

    Returns:
        config_file_dict (dict) : A dictionary of the default configuration
            files.
    """

    config_file_dict = {
        "template_toml_file": "src/us_templates.toml",
        "patient_id_config": "src/tesseract_config/patient_id.txt",
        "gest_day_config": "src/tesseract_config/gest_days.txt",
        "dateofexam_config": "src/tesseract_config/dateofexam.txt",
        "vessel_type_config": "src/tesseract_config/vessel_type.txt",
    }

    return config_file_dict


# Increase DPI
def increase_dpi(image, factor=2):
    """Increases the DPI of an image by a factor

    Args:
        image (ndarray) : A 2D numpy array of the image.
        factor (int, optional) : The factor to increase the DPI by.

    Returns:
        col_image (ndarry) : A 2D numpy array of the image with increased DPI.
    """

    # Increases the number of rows
    row_image = np.zeros((image.shape[0] * factor, image.shape[1]))
    for i in range(image.shape[0]):
        new_rows = np.arange(i * factor, (i + 1) * factor)
        row_image[new_rows, :] = image[i, :]

    # Increases the number of cols
    col_image = np.zeros((row_image.shape[0], row_image.shape[1] * factor))
    for j in range(row_image.shape[1]):
        new_cols = np.arange(j * factor, (j + 1) * factor)
        col_image[:, new_cols] = row_image[:, j][..., None]

    return col_image


def gest_str_to_days(gest_day_str):
    """Converts gestation days string extracted from ultrasound into an integer of days

    Returns -1 if no text was extracted.
    Fails if could not read extracted text.

    Args:
        gest_day_str (str) : A string of the format 'NwM',
            where N is the number of weeks and M is the number of days.

    Returns:
        gest_day_int (int) : The number of gestation days as an integer.
    """

    # Case for no text extracted
    if gest_day_str in ("N/A", ""):
        logger.warning(
            "No gestation days text extracted.\n"
            "Gestation days probably weren't recorded.\n"
            "Returning -1."
        )

        return -1

    # Calculates the number of gestation days from days and weeks
    try:
        weeks, days = gest_day_str.lower().split("w")
    except ValueError:
        logger.error(f"Couldn't extract gestation day from {gest_day_str}")
        return -1

    weeks = weeks.strip("=:")
    days = days.strip("d")
    gest_day_int = int(np.abs(int(weeks)) * 7 + np.abs(int(days)))

    return gest_day_int


def get_labelled_vessel_type(vessel_text, no_text_extracted="N/A"):
    """Gets the labelled vessel type from image text

    Args:
        vessel_text (str) : The vessel text extracted from the image.
            Expected inputs are "L", "U" and "R".

    Returns:
        vessel_type (int) : Either 0, 1 or 2 for umbilical, left and right uterine respectively.
    """

    if len(vessel_text) > 1 and vessel_text != no_text_extracted:
        logger.debug(f"Assuming {vessel_text} as {vessel_text[0]}")
        vessel_text = vessel_text[0]

    if vessel_text.upper() == "U":
        vessel_type = 0
    elif vessel_text.upper() == "L":
        vessel_type = 1
    elif vessel_text.upper() == "R":
        vessel_type = 2
    elif vessel_text == no_text_extracted:
        # There a lots of images with no vessel type so this should
        # be a debug message rather than a warning.
        logger.debug("Vessel type not extracted from image.")
        vessel_type = -1
    else:
        # This however, is probably worth looking into
        logger.warning(f"Could not determine vessel type from {vessel_text}\n")
        vessel_type = -2

    return vessel_type


def get_text_from_us(
    image,
    config="--psm 7 --oem 3",
    config_file="src/tesseract_config/patient_id.txt",
    ext="txt",
    scale_factor=2,
):
    """Reads textual metadata of an image

    Args:

        image (object) : A cv2 image object in RGB format.
        config (str, optional) : The configuration to be passed to the tesseract library.
            Defaults to "--psm 7 --oem 3".
        config_file (str, optional) : The relative path to the tesseract configuration file.
            Defaults to 'src/tesseract_config/patient_id.txt'.
        ext (str, optional) : The extension of the output type.
            Defaults to "txt".
        scale_factor (int, optional) : The factor to scale up the image resolution by.
            For some reason 2 works well.
            Defaults to 2.

    Returns:
        image_str (str) : A string of the image text
    """

    # Increases the DPI of the image and performs a Gaussian filter
    image_dpi = increase_dpi(image[:, :, 0], factor=scale_factor)
    image_final = np.zeros((image_dpi.shape[0], image_dpi.shape[1], image.shape[2]))
    image_final += skimage.filters.gaussian(image_dpi, sigma=1)[..., None]
    image_final = image_final.astype(np.uint8)

    # Performs binarisation of image
    threshold = np.max(image_final) / 2
    image_bin = np.zeros(image_final.shape)
    image_bin[image_final < threshold] = 255
    image_final = np.copy(image_bin.astype(np.uint8))

    # Creates the configuration string to be passed to tesseract
    config_file_full_path = os.path.join(os.getcwd(), config_file).replace('\\','/')
    tessconfig = f"{config} configfile {config_file_full_path}"

    # Gets the text from the image
    image_str = pytesseract.run_and_get_output(
        image_final,
        extension=ext,
        config=tessconfig,
        timeout=0,
    )

    if len(image_str.split()) == 0:
        logger.warning("Empty string extracted. Setting to 'N/A'.")
        image_str = "N/A"
    else:
        image_str = image_str.split()[0]

    return image_str


def scale_image_template(image, template, skip_keys=None):
    """Scales a template to the current image size

    Args:
        image (ndarry) : A numpy array of the image.
        template (dict) : A template to scale to the size of the image.
        skip_keys (list) : Keys to skip scaling.
            If None, defaults to ["name", "key_str"].
            Defaults to None.

    Returns:
        new_template (dict) : A scaled template to the image
    """

    # Checks skip_keys
    if isinstance(skip_keys, type(None)):
        skip_keys = ["name", "key_str"]

    # Gets the scaled image coordinates to account for the same template
    # but different resolutions
    if np.all(image.shape[:2] != template["image_size"]):
        # Scale factor for x and y
        x_scale = image.shape[0] / template["image_size"][0]
        y_scale = image.shape[1] / template["image_size"][1]

        for key in list(template.keys()):
            if key not in skip_keys:
                # Converts to numpy array
                template[key] = np.array(
                    template[key],
                    dtype=float,
                )
                # Template key
                template[key][:2] *= x_scale
                template[key][2:] *= y_scale

                # Converts to ints
                template[key] = np.array(
                    template[key],
                    dtype=np.uintc,
                )

    return template


# Gets the institute template from a list of templates and an image
def determine_image_template(
    image, template_list, config_file="src/tesseract_config/NONE.txt"
):
    """Determines returns the appropriate template from a list of templates

    Args:
        image (ndarray) : A numpy array of the ultrasound image.
        template_list (tuple) : A list of institute templates.

    Returns:
        institute_template (dict) : The ultrasound image template for the institute.
    """

    # Iterates over image templates to check to see if image matches
    keys = []
    for template in template_list:
        template = scale_image_template(image, template)

        # Crops the image and extracts text
        image_key = image[
            template["key_coords"][0] : template["key_coords"][1],
            template["key_coords"][2] : template["key_coords"][3],
        ]
        key = get_text_from_us(image_key, config_file=config_file)

        # If the key matches the template, return that template
        if key == template["key_str"].split()[0]:
            return template

        # Appends keys doesn't match that of the template
        keys.append(key)

    logger.error(
        "Could not identify image institute.\n"
        f"The following keys were obtained:\n{keys}"
    )
    raise ValueError("Unknown Image Institute")


# Gets the ultrasound metadata
def get_us_metadata(
    image,
    config_file_dict=None,
    date_sep="/",
):
    """Gets a complete set of metadata from the image.

    Args:
        image (ndarry) : A numpy array to be passed to tesseract.
        config_dict (dict) : A dictionary containing the paths to the
            configuration files. Defaults to None. If None, fills dictionary
            with default structure. Structure:
            template_toml_file (str, optional) : The TOML file containing the
                institute ultrasound templtes.
                Defaults to 'src/us_templates.toml'.
            patient_id_config (str, optional) : The tesseract configuration file
                for the patient ids.
                Defaults to 'src/tesseract_config/patient_id.txt'.
            gest_day_config (str, optional) : The tesseract configuration file
                for the gestation days.
                Defaults to 'src/tesseract_config/gest_day.txt'.
            dateofexam_config (str, optional) : The tesseract configuration file
                for the date of exam.
                Defaults to 'src/tesseract_config/dateofexam.txt'.
            vessel_type_config (str, optional) : The tesseract configuration
                file for the vessel type.
            Defaults to 'src/tesseract_config/vessel_type.txt'.
        date_sep (str, optional) : What to replace the date joining character with.
            Defaults to "/".

    Returns:
        metadata (dict) : A dictionary containing the extracted metadata.
    """

    if config_file_dict is None:
        config_file_dict = load_default_config_file_dict()

    # Determines the image institute
    template_list = load_us_templates(config_file_dict["template_toml_file"])
    institute_template = determine_image_template(image, template_list)

    def _crop_and_extract_text(coords, config_file, factor=2):
        """Crops and extracts the text from the image."""

        # Crops the image
        cropped_image = image[
            coords[0] : coords[1],
            coords[2] : coords[3],
        ]

        # Extracts the text
        extracted_text = get_text_from_us(
            cropped_image, config_file=config_file, scale_factor=factor
        )

        return extracted_text

    # Extracts the text
    patient_id_text = _crop_and_extract_text(
        institute_template["patient_id_coords"],
        config_file_dict["patient_id_config"],
    )
    gest_day_text_str = _crop_and_extract_text(
        institute_template["gest_day_coords"],
        config_file_dict["gest_day_config"],
    )
    gest_day_text = gest_str_to_days(gest_day_text_str)
    dateofexam_text = _crop_and_extract_text(
        institute_template["dateofexam_coords"],
        config_file_dict["dateofexam_config"],
    )
    dateofexam_text = date_sep.join(dateofexam_text.split("."))
    vessel_type_text_raw = _crop_and_extract_text(
        institute_template["vessel_type_coords"],
        config_file_dict["vessel_type_config"],
        factor=4,
    )
    vessel_type_text = get_labelled_vessel_type(vessel_type_text_raw)

    # Puts everything in a dictionary
    metadata = {
        "patient_id": patient_id_text,
        "gest_day": gest_day_text,
        "dateofexam": dateofexam_text,
        "vessel_type": vessel_type_text,
    }

    return metadata


def segment_doppler(cv2_img):
    """Extract textual information and signal from ultrasound doppler.

    Wraps around the data_from_image function from the usseg library.

    Args:
        cv2_img (ndarray) : Numpy array of the image.

    Returns:
        singal (tuple) : A tuple containing a numpy array for the time axis and
                signal respectively.
    """

    pil_img = Image.fromarray(cv2_img)

    df, (t_axis, signal) = usseg.data_from_image(pil_img, cv2_img)
    t_axis = np.array(t_axis, ndmin=2)
    signal = np.array(signal, ndmin=2)
    return df, t_axis, signal
