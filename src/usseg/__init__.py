"""Initialises the ultrasound-segmentation module"""
from importlib.metadata import version, PackageNotFoundError
from usseg import general_functions
from usseg.organise_files import get_likely_us
from usseg.single_image_processing import data_from_image
from usseg.segment_files import segment
from usseg.setup_environment import setup_tesseract
from usseg.visualisation_html import generate_html_from_pkl, generate_html
from usseg.main import main
# Replace this with loading from configuration.
# import os
# os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

try:
    __version__ = version("usseg")
except PackageNotFoundError:
    # package is not installed
    __version__ = "0.0.0"
