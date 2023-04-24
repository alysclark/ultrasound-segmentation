import os
import re
import pickle
import traceback
import General_functions
"""
This script searches a given directory and identifies the images that are likely to be doppler ultrasounds.
"""
# root diretory to search
root_dir = "E:/us-data-anon/0000"

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
                        # print(Fail)
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
                            # print(Fail)
                            if Fail == 0:
                                # Append the path to the list for this patient
                                patient_path_list.append(os.path.join(subdir, file))
                        except Exception:
                            traceback.print_exc()  # prints the error message and traceback
                            continue
            # Add the list of paths to the dictionary for this patient
            patient_paths[patient_id] = patient_path_list

# save the patient_paths dictionary to a file in current directory
with open('patient_paths_test.pkl', 'wb') as f:
    pickle.dump(patient_paths, f)
