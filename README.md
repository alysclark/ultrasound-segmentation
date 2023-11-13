# Doppler Segmentations

These codes make up the framework for segmenting the doppler ultrasound scans.

## Table of Contents

- [Doppler Segmentations](#doppler-segmentations)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
- [Ultrasound Segmentation Package](#ultrasound-segmentation-package)


## Installation

- To install as a dependency for your project:

``` 
pip install usseg
```

or

``` sh
poetry add usseg
```


### Development Environment

To install the development environment follow the following steps.

- Clone this repository and change into the directory.
- Install [poetry](https://python-poetry.org/docs/) as per the installation instructions.
- Install [tesseract](https://github.com/tesseract-ocr/tesseract) as per the intallation instructions.
    - Note that the project has only been tested with tesseract version 5.
- Install the package dependencies with:
```
poetry install

```
- Enter into the development shell with:

```
poetry shell
```

- You are now in the development environment!
- Copy `config_example.toml` to `config.toml` and change the variables for your local set up (e.g. path to your data etc.).
- The main script can now be run in one complete run with `python usseg/main.py`.
- If debugging in vscode, ensure the python interpreter is set to the virtual environment created poetry. This path can be found using ```poetry env info --path```


# Ultrasound Segmentation Package

The Ultrasound Segmentation Package facilitates two primary blocks of functionality: 
text extraction from images and ultrasound image segmentation. Each block consists 
of a sequence of functions designed to work independently yet contribute collectively 
to the overall process.

### Text Extraction Process

1. **colour_extract_vectorized**: Filters the image to highlight a specific target colour of pixel, preparing it for text extraction.
<div align="center">
   <img src="Documentation/source/Vectorized_colour_extraction_diagram.png" width="45%" alt="Colour Extract Vectorized" />
</div>

2. **Text_from_greyscale**: Processes the filtered image to extract text, matching lines to specific target words.
<div align="center">
   <img src="Documentation/source/Text_extraction_diagram.png" alt="Text Extraction" width="45%"/>
</div>

3. **Metric_check**: Performs a common-sense check on the extracted measurements using knowledge of their interdependencies and known physiological limits, ensuring data accuracy.
<div align="center">
   <img src="Documentation/source/df_data_extracted_diagram.png" width="45%" alt="Data Extracted"/>
</div>
Following the successful extraction and validation of text data, the workflow transitions to the image segmentation process.

### Image Segmentation Process

4. **Initial_segmentation**: Begins with a coarse segmentation of the waveform.
5. **Define_end_ROIs**: Defines regions adjacent to the coarse segmentation.
<div align="center">
   <img src="Documentation/source/Initial_segmentation_diagram.png" alt="Initial Segmentation"/>
</div>

6. **Segment_refinement**: Refines the segmentation within the coarse boundaries.
<div align="center">
   <img src="Documentation/source/Segment_refinement_diagram.png" alt="Segment Refinement"/>
</div>

7. **Search_for_ticks**: Identifies ticks in the axes ROIs for accurate scaling.
8. **Search_for_labels**: Locates labels within the axes ROIs for data extraction.

<div align="center" style="display: flex; justify-content: center;">
    <img src="Documentation/source/TickandLabel_diagram.png" width="35%" alt="Search for Ticks" style="margin-right: 30px;"/>
    <img src="Documentation/source/ROIAX_change_diagram.png" width="35%" alt="Search for Labels"/>
</div>

9. **Plot_Digitized_data**: Digitizes the extracted data to plot the waveform.
<div align="center">
   <img src="Documentation/source/Digitize_Function_diagram.png" alt="Plot Digitized Data"/>
</div>

10. **Plot_correction**: (Optional) Adjusts the time axis based on heart rate data.
11. **Annotate**: Visualizes the segmentation steps on the original image.
<div align="center">
    <img src="Documentation/source/Overview2.png" alt="Annotate"/>
</div>

Each function in these sequences plays a vital role in the overall process, which aims for accurate data extraction. For more in-depth information about each function, please refer to the detailed descriptions in the [here](usseg.html) section of this documentation.

## Usage Examples

Some introduction to examples...

### Processing a Single Image

For processing a single image, the `data_from_image` function is imported and provided with PIL and cv2 versions of the image. This could be done through the following code:

```python
# Module imports
import numpy as np
from PIL import Image

# Local imports
from usseg import data_from_image

img_path = "Path/to/a/ultrasound/image.JPG"

PIL_image = Image.open(img_path)
cv2_image = np.array(PIL_image)
df, (xdata, ydata) = data_from_image(PIL_image, cv2_image)
```
### Batch processing images
```python
python usseg/main.py
```