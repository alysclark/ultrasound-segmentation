Doppler Segmentations
=====================

These codes make up the framework for segmenting the doppler ultrasound scans.

.. contents::
   :local:
   :depth: 2

Installation
------------

To install as a dependency for your project:

.. code-block:: bash

    pip install usseg

or

.. code-block:: sh

    poetry add usseg

Development Environment
-----------------------

To install the development environment follow the following steps:

- Clone this repository and change into the directory.
- Install `poetry` as per the installation instructions.
- Install `tesseract` as per the installation instructions.
  - Note that the project has only been tested with tesseract version 5.
- Install the package dependencies with:

.. code-block:: bash

    poetry install

- Enter into the development shell with:

.. code-block:: bash

    poetry shell

- You are now in the development environment!
- Copy ``config_example.toml`` to ``config.toml`` and change the variables for your local set up (e.g., path to your data, etc.).
- The main script can now be run in one complete run with:

.. code-block:: bash

    python usseg/main.py

- If debugging in VSCode, ensure the Python interpreter is set to the virtual environment created by poetry. This path can be found using:

.. code-block:: bash

    poetry env info --path

