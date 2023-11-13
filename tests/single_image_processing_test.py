"""Test the single image processing function."""

# Python imports
import os
import sys
import logging
import pytest

# Module imports
import numpy as np
from PIL import Image

# Local imports
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)
from usseg import data_from_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)


def test_data_from_image():
    """Test the data_from_image function."""
    img_path = "E:/us-data-anon/0000\IHE_PDI/00003511\AA13981B\AA196E50/00009020/EE277D8A.JPG"

    PIL_image = Image.open(img_path)
    cv2_image = np.array(PIL_image)
    logger.info(f"Loaded image with shape {cv2_image.shape} and type {cv2_image.dtype}")

    df, (xdata, ydata) = data_from_image(PIL_image, cv2_image)

    # Makes sure that the lists aren't empty
    assert xdata
    assert ydata

    logger.info(f"Extracted the following text from {img_path}:\n{df}")


def test_failures():
    """Test that the correct fail responses are being raised."""
    
    # Failed extraction
    cv2_img = np.random.default_rng().integers(0, 256, size=(100, 100, 3), dtype=np.uint8)
    PIL_img = Image.fromarray(cv2_img)

    with pytest.raises(ValueError) as exc_info:
        data_from_image(PIL_img, cv2_img)

    exc_raised = str(exc_info.value)
    assert exc_raised == "attempt to get argmax of an empty sequence"


if __name__ == "__main__":
    test_data_from_image()
    #test_failures()
    logger.info(f"{__file__} tests have passed!")
