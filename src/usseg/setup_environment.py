import logging
import pytesseract
import sys

logger = logging.getLogger(__file__)


def setup_tesseract():
    """Checks tesseract is set up appropriately

    Currently, does nothing on a linux system and sets the
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
