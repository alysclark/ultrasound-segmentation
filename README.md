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

Describe the installation process for your project.

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

Digitized the ticks and labels to plot waveform.

## Plot_correction

If avaliable, corrects the x-axis from arbitrary time units to seconds based on extracted heart rate.

## Annotate

Function for visualising the segmentation steps by annotating the original image with the segmenented wavefore, ticks and labels identified from the previous functions.

## Contributing

Explain how others can contribute to your project.

## License

Add information about the license for your project, and any relevant copyright information.
