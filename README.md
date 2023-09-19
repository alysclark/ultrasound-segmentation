# Doppler Segmetnations

These codes make up the framework for segmenting the doppler ultrasound scans.

## Table of Contents

- [Doppler Segmetnations](#doppler-segmetnations)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
- [Functions](#functions)
  - [Initial\_segmentation](#initial_segmentation)
  - [Define\_end\_ROIs](#define_end_rois)
  - [Segment\_refinement](#segment_refinement)
  - [Search\_for\_ticks](#search_for_ticks)
  - [Search\_for\_labels](#search_for_labels)
  - [Plot\_Digitized\_data](#plot_digitized_data)
  - [Plot\_correction](#plot_correction)
  - [Annotate](#annotate)
  - [Contributing](#contributing)
  - [License](#license)

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


# Functions

Provide instructions on how to use your project.

## Initial_segmentation

Preform an initial corse segmentation of the waveform.

## Define_end_ROIs

Function to define regions adjacent to the corse waveform.
![alt text](Initial_segmentation_diagram.png)

## Segment_refinement

Function to refine the waveform segmentation within the bounds of the corse waveform ROI.

![alt text](Segment_refinement_diagram.png)
## Search_for_ticks

Function to search for any potential ticks in either of the axes ROIs, also crops the ROI for each axes to avoid the ticks, which can interfere with tesseract text detection in the next function.

![alt text](ROIAX_change_diagram.png)

## Search_for_labels

Function that searches for labels withing the axes ROI.

![alt text](TickandLabel_diagram.png)


## Plot_Digitized_data

A function for digitizing the extracted ticks and labels data to plot a waveform, using the output from *segment refinement*. This requries the 

![alt text](Digitize_Function_diagram.png)

## Plot_correction

If avaliable, corrects the x-axis from arbitrary time units to seconds based on extracted heart rate.

## Annotate

Function for visualising the segmentation steps by annotating the original image with the segmenented wavefore, ticks and labels identified from the previous functions.

## Contributing

Explain how others can contribute to your project.

## License

Add information about the license for your project, and any relevant copyright information.
