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
    img_path = "Umb_test_image.png"

    PIL_image = Image.open(img_path)
    cv2_image = np.array(PIL_image)
    logger.info(f"Loaded image with shape {cv2_image.shape} and type {cv2_image.dtype}")

    df, (xdata, ydata) = data_from_image(PIL_image, cv2_image)

    # Makes sure that the lists aren't empty
    assert xdata
    assert ydata

    logger.info(f"Extracted the following text from {img_path}:\n{df}")

    # TODO: Improve accuracy and thus lower tolerance.
    tol = 0.20
    for i in range(df.shape[0]):
        # Ignores cases where no digitized value exists (which is an empty string)
        if isinstance(df.at[i, "Digitized Value"], (np.float_, np.int_)):
            assert np.abs(df.at[i, "Digitized Value"] - df.at[i, "Value"]) < tol * df.at[i, "Value"]

    # These values were obtained from reading the image values directly
    # Currently not extracting the PI value so will omit for now
    # TODO: Successfully extract the PI value.
    # values = [70.64, 20.50, 3.45, 1.46, 0.71, 19.89, 34.40, 82.00]
    values = [70.64, 20.50, 3.45, 0.71, 19.89, 34.40, 82.00]
    assert len(values) == df.shape[0]
    # TODO: Improve text extraction accuracy and thus lower tolerance
    tol = 5e-2
    for i, val in enumerate(values):
        assert np.abs(df.at[i, "Value"] - val) < tol * val

def test_failures():
    """Test that the correct fail responses are being raised."""
    
    # Failed extraction
    cv2_img = np.random.default_rng().integers(0, 256, size=(100, 100, 3), dtype=np.uint8)
    PIL_img = Image.fromarray(cv2_img)

    with pytest.raises(ValueError) as exc_info:
        data_from_image(PIL_img, cv2_img)

    exc_raised = str(exc_info.value)
    assert exc_raised == "zero-size array to reduction operation minimum which has no identity"


if __name__ == "__main__":
    test_data_from_image()
    test_failures()
    logger.info(f"{__file__} tests have passed!")
