"""Searches a directory and identifies images likely to be doppler ultrasounds"""
# /usr/bin/env python3

# Python imports
import os
import sys
import re
import pickle
import traceback

# Module imports
import toml

# Local imports
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)
from usseg import General_functions

def get_likely_us(root_dir, pickle_path=None):
    """Searches a directory and identifies the images that are likely to be doppler ultrasounds.

    Args:
        root_dir (str) : Path to the root directory .
        pickle_path (str or bool) : If pickle_path is False, will not store the
            list of likely us images as a pickle file. If None,
            will load the pickle path from "config.toml".
            Else if a string, will dump the pickled list to the specified path.
            Defaults to None.

    Returns:
        return_val (bool) : True if sucessful, False otherwise.
    """
# root diretory to search

    # Initialize a dictionary to store the paths for each patient
    patient_paths = {}

    for patient_dir in os.listdir(root_dir):  # If the root_dir is to a folder containing multiple patients
        # Check if the directory name is a patient ID (4 digits)
        if len(patient_dir) == 4 and patient_dir.isdigit():
            patient_id = patient_dir
            # Initialize a list to store the paths for this patient
            patient_path_list = []

            # Traverse the patient directory
            patient_dir_path = os.path.join(root_dir, patient_dir)
            for subdir, dirs, files in os.walk(patient_dir_path):
                for file in files:
                    if file.endswith('.JPG'):
                        try:
                            Fail, df = General_functions.Scan_type_test(os.path.join(subdir, file))
                            if Fail == 0:
                                # Append the path to the list for this patient
                                patient_path_list.append(os.path.join(subdir, file))
                        except Exception:
                            traceback.print_exc()  # prints the error message and traceback
                            continue
            # Add the list of paths to the dictionary for this patient
            patient_paths[patient_id] = patient_path_list
        else:  # if the root_dir is to a specific patient folder
            match = re.search(r"\d{4}", root_dir)
            if match:
                patient_id = match.group(0)
                # Initialize a list to store the paths for this patient
                patient_path_list = []

                # Traverse the patient directory
                patient_dir_path = os.path.join(root_dir, patient_dir)
                for subdir, dirs, files in os.walk(patient_dir_path):
                    for file in files:
                        if file.endswith('.JPG'):
                            try:
                                Fail, df = General_functions.Scan_type_test(os.path.join(subdir, file))
                                if Fail == 0:
                                    # Append the path to the list for this patient
                                    patient_path_list.append(os.path.join(subdir, file))
                            except Exception:
                                traceback.print_exc()  # prints the error message and traceback
                                continue
                # Add the list of paths to the dictionary for this patient
                patient_paths[patient_id] = patient_path_list

    # save the patient_paths dictionary to a file in current directory
    if pickle_path is not False:
        if pickle_path is None:
            pickle_path = toml.load("config.toml")["pickle"]["likely_us_images"]

        with open(pickle_path, 'wb') as f:
            pickle.dump(patient_paths, f)

    return pickle_path
    
if __name__ == "__main__":

    root_dir = toml.load("config.toml")["root_dir"]
    get_likely_us(root_dir)
