# Standard library imports
import itertools
import os
from typing import Dict, Optional

# Third-party imports
import cv2
import numpy as np
import pandas as pd
import tkinter as tk
import yaml
from PIL import Image, ImageTk
from tqdm import tqdm

# Application-specific imports
from model_preprocessing import model_preprocessing
from testing_and_visualize_map_preprocessing import *
from utils_preprocessing import *


def evaluate_params(filenames_imgs=None, show_demonstration: Optional[bool] = False, **kwargs):
    """
    Rotate randomly the images and try to rotate the image back. The Loss is the difference between validated
    rotation angle and predicted rotation angle per image.
    Arguments: parameters to test
    Return:
        Loss: difference between validated rotation angle and predicted rotation angle per image
    """

    # create 12 randomly rotated images of the folder "data\benchmark_maps\prepared_for_testing"
    # rotated_imgs[0] -> filename, rotated_imgs[1] -> image, rotated_imgs[2] -> rotated angle
    rotated_imgs = create_test_imgs(filenames_imgs)

    # Loop through the test set of images
    diffs = []
    preprocessed_list = []
    for img in rotated_imgs:
        # preprocess the test images
        preprocessed = model_preprocessing(img[1], **kwargs)
        preprocessed_list.append(preprocessed)
        # calculate the difference between validated rotation angle and predicted rotation angle
        diff = round(abs(img[2] - preprocessed.largest_cluster_of_the_orientations), 1)
        if diff > 45.0:  # Ensure maximum error is 45 degrees
            diff = round(90 - diff, 1)
        diffs.append(diff)

    # calculate the loss
    loss = sum(diffs)

    if show_demonstration:
        # show all images with their respective errors and line counts
        preprocessed_imgs = [preprocessed.rotated_image for preprocessed in preprocessed_list]
        diffs_and_preprocessed = [(diff, preprocessed) for diff, preprocessed in zip(diffs, preprocessed_list)]
        diffs = [
            f"Fehler: {diff}, Linien erkannt: {len(preprocessed.angles_of_walls)}" for diff, preprocessed in diffs_and_preprocessed]
        show_all_imgs(preprocessed_imgs, diffs)
        print(f"Score aller Bilder: {loss}")
        print(f"Im durchschnitt ist ein Bild um {loss/len(rotated_imgs)} verdreht")
    return loss/len(rotated_imgs)


def param_dict_to_combinations_in_df(param_dict: dict) -> pd.DataFrame:
    """
    Converts a dictionary of parameters with lists of possible values into a DataFrame of all possible combinations.

    Arguments:
        param_dict: a dictionary where keys are parameters and values are lists of possible values

    Returns:
        A DataFrame where each row is one combination of parameter values
    """
    # Get all possible combinations of parameter values using itertools.product
    # The * operator is used to unpack the dictionary values (lists of possible values)
    value_combinations = itertools.product(*param_dict.values())

    # For each combination, create a dictionary where the keys are the parameter names
    # and the values are the values for this combination. Collect these dictionaries into a list.
    dict_list = [dict(zip(param_dict.keys(), values)) for values in value_combinations]

    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(dict_list)

    return df


def test_all_param_combinations(param_dict: Dict[str, list], filenames_imgs=None,  evaluation_runs: int = 10, show_demonstration: Optional[bool] = False) -> pd.DataFrame:
    """
    Test all combinations of parameters and calculate the average loss.

    Arguments:
        param_dict: contains the names of the parameters as keys and the values to be tested as lists in the values
        evaluation_runs: number of times to run the evaluation for each parameter set. To reduce Randomness of the random rotation angle

    Returns:
        df: a dataframe containing all combinations of all possible parameter combinations with their corresponding loss
    """
    df = param_dict_to_combinations_in_df(param_dict)
    df["loss"] = np.nan  # initialize loss column

    # Create a progress bar
    pbar = tqdm(total=len(df))

    # Iterate through the dataframe
    for index, row in df.iterrows():
        # Run the evaluation for each set of parameters and evalutation_runs
        total_loss = sum(
            evaluate_params(filenames_imgs,
                            show_demonstration,
                            kernel_size=int(row["kernel_size"]),
                            rho=row["rho"],
                            theta=row["theta"],
                            threshold=int(row["threshold"]),
                            minLineLength=row["minLineLength"],
                            maxLineGap=row["maxLineGap"],
                            eps=row["eps"],
                            min_samples=int(row["min_samples"]),
                            show_img=False)
            for _ in range(evaluation_runs)
        )

        # Calculate the average loss
        avg_loss = round(total_loss / evaluation_runs, 2)

        # write the calculated loss in the loss column and the current row
        df.at[index, 'loss'] = avg_loss

        # for testing the right allocation
        for param in param_dict.keys():
            df.at[index, f'tested_{param}'] = row[str(param)]

        # Update progress bar
        pbar.update(1)

    pbar.close()

    df = test_loss_allocation_in_df(df, param_dict)

    return df


def write_df_to_csv(df, base_filepath):
    """
    Writes the DataFrame to a CSV file.

    Parameters:
    df (pandas.DataFrame): DataFrame to be written to a CSV file.
    base_filepath (str): The base filepath for the CSV file.
                         The version number and '.csv' extension will be appended to this.

    Returns:
    str: The filepath of the written CSV file.
    """

    # Create the filepath for the new version
    filepath = f'{base_filepath}.csv'

    # Write the DataFrame to a CSV file
    df.to_csv(filepath, index=False)

    return filepath


def test_loss_allocation_in_df(df, params):
    """
    Check if tested parameters are equal to the original parameters in each row of a dataframe.

    Arguments:
        df: pandas DataFrame to be checked.
        params: dictionary of parameters to be checked.

    """
    for param in params.keys():
        df[f'{param}_allocation'] = df[f'tested_{param}'] == df[param]
    return df


def read_out_the_best_param_combination_of_df(df):
    """
    Extracts and returns the best parameter combination from a DataFrame based on minimum loss.

    This function finds the row in the DataFrame with the minimum 'loss' value, and then extracts
    the associated parameter values. It assumes that the DataFrame has the following columns:
    'kernel_size', 'rho', 'theta', 'threshold', 'minLineLength', 'maxLineGap', 'eps', and 'min_samples'.

    Parameters:
    df (pandas.DataFrame): DataFrame containing parameter combinations and associated loss values.
                           Each row represents a unique combination of parameters and the resulting loss.

    Returns:
    dict: Dictionary containing the best parameter combination. Each key is a parameter name and the
          associated value is the optimal parameter value (from the row with the minimum loss).
    """
    # Find the row with the minimum loss
    min_loss_row = df.loc[df['loss'].idxmin()]

    # Extract the parameter values from the row
    best_params = {
        'kernel_size': min_loss_row['kernel_size'],
        'rho': min_loss_row['rho'],
        'theta': min_loss_row['theta'],
        'threshold': min_loss_row['threshold'],
        'minLineLength': min_loss_row['minLineLength'],
        'maxLineGap': min_loss_row['maxLineGap'],
        'eps': min_loss_row['eps'],
        'min_samples': min_loss_row['min_samples']
    }

    return best_params


def write_params_to_yaml(params, filename):
    """
    Writes a dictionary of parameters into a YAML file.

    This function takes a dictionary of parameters and writes it into a YAML file.
    The resulting file will have each parameter on a new line with the parameter
    name followed by a colon and the parameter value.

    Parameters:
    params (dict): Dictionary containing parameter combinations.
                   Each key is a parameter name and the associated value is the optimal parameter value.
    filename (str): The name of the YAML file to which the parameters should be written.

    Returns:
    None
    """
    # Convert NumPy types to native Python types
    params = {k: v.item() if isinstance(v, np.generic) else v for k, v in params.items()}

    with open(filename, 'w') as file:
        yaml.dump(params, file, default_flow_style=False)


def load_csv_as_dataframe(file_path):
    """
    Load a CSV file and return it as a pandas DataFrame.

    Parameters:
        file_path (str): The path to the CSV file.

    Returns:
        DataFrame: The loaded data as a DataFrame.
    """
    df = pd.read_csv(file_path)
    return df


def select_imgs_for_tuning(path):
    imgs = read_in_all_images_from_file(path)

    selected_filenames_and_imgs = []
    photo_images = []  # list to hold PhotoImage objects to prevent garbage collection

    def make_on_click(img_label, img_tuple):
        def on_click(event):
            if img_tuple not in selected_filenames_and_imgs:
                selected_filenames_and_imgs.append(img_tuple)
                img_label.config(borderwidth=2, relief="solid")
        return on_click

    def close_window():
        root.destroy()

    # Create a new tkinter window
    root = tk.Tk()

    # Add a button to close the window
    exit_button = tk.Button(root, text="Finish Selection", command=close_window)
    exit_button.pack()

    # Add a canvas within window which will contain the images.
    canvas = tk.Canvas(root)
    canvas.pack(side="left", fill="both", expand=True)

    # Add a scrollbar to the right side of the window.
    scrollbar = tk.Scrollbar(root, command=canvas.yview)
    scrollbar.pack(side="left", fill="y")

    # Configure the canvas to connect with the scrollbar.
    canvas.configure(yscrollcommand=scrollbar.set)

    # Add a frame within the canvas which will hold the grid of labels with images.
    frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=frame, anchor="nw")

    # Parameters for grid layout
    images_per_row = 4
    row_index = 0
    column_index = 0

    for img_tuple in imgs:
        # Extract the image from the tuple
        img = cv2.cvtColor(img_tuple[1], cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        img_pil = Image.fromarray(img)
        img_pil.thumbnail((400, 400))  # resize the image so that it fits within a 400x400 square
        img_tk = ImageTk.PhotoImage(img_pil)
        photo_images.append(img_tk)  # store a reference to prevent garbage collection

        # Create a label for the image
        img_label = tk.Label(frame, image=img_tk)
        img_label.grid(row=row_index, column=column_index)

        # Bind the mouse click event to the label
        img_label.bind("<Button-1>", make_on_click(img_label, img_tuple))

        # Update row and column indices
        column_index += 1
        if column_index >= images_per_row:
            column_index = 0
            row_index += 1

    # Update scroll region after tkinter completes drawing the window
    frame.update_idletasks()
    canvas.configure(scrollregion=canvas.bbox('all'))

    # Run the tkinter main loop
    root.mainloop()

    # Return the selected images after the main loop is closed
    return selected_filenames_and_imgs


def get_input(default_value):
    value = input(f'Enter a value (default is {default_value}): ')
    if value == '':
        return default_value
    else:
        return value


def optimize_parameter_pipeline(params, path_to_imgs="data\\benchmark_maps\\prepared_for_testing", evaluation_runs=10):
    """Optimizes parameters for image rotation and writes the best parameters to a yaml file."""

    # select the images on which it should be optimized
    selected_filenames_and_imgs = select_imgs_for_tuning(path_to_imgs)
    # takes parameter with ranges
    print(f"It will be optimized on {len(selected_filenames_and_imgs)} images")
    print("How many evaluation runs do you want to conduct? This number represents how many times an image will be rotated by a random degree, and the loss will be calculated on the re-rotated image.")

    # Test all combinations of parameters and calculate the average loss. Returns a dataframe containing all combinations of all possible parameter combinations with their corresponding loss
    user_input = int(get_input(evaluation_runs))
    df = test_all_param_combinations(params, selected_filenames_and_imgs, user_input)

    # Find the version number for the output file
    version = 1
    while os.path.isdir(f'parameter_tuning\\tested_parameters_version_{version}'):
        version += 1

    # Create the filepath for the new version
    folderpath = f'parameter_tuning\\tested_parameters_version_{version}'
    os.makedirs(folderpath, exist_ok=True)

    base_filepath = f"{folderpath}\\parameters"
    # Save the test as CSV and return the filepath
    filepath = write_df_to_csv(df, base_filepath)
    # Split the filepath into a name and extension, and only return the name
    filepath_without_extension, _ = os.path.splitext(filepath)

    """Writes filenames of images in a .txt"""
    with open(f"{filepath_without_extension}_filenames_of_images.txt", "w") as file:
        for selected_filenames_and_img in selected_filenames_and_imgs:
            filename = selected_filenames_and_img[0]
            file.write(filename + "\n")

    # Extracts and returns (=dictionary) the best parameter combination from a DataFrame based on minimum loss
    best_params = read_out_the_best_param_combination_of_df(df)
    print(f"The combination, that reached the lowest loss is: {best_params}")

    # Save the best params as yaml
    yaml_filepath = filepath_without_extension + str("_param_combination_with_lowest_loss")
    write_params_to_yaml(best_params, yaml_filepath)
    print(f"Best YAML-File was saved in {yaml_filepath}")


if __name__ == '__main__':
    param_dict = {
        "kernel_size": [3, 5],
        "rho": [0.8, 16.6],
        "theta": [np.pi/360, np.pi/160],
        "threshold": [30, 60],
        "minLineLength": [40, 70],
        "maxLineGap": [2, 4],
        "lengths_divisor": [20],
        "eps": [0.5, 1, 1.5],
        "min_samples": [10, 30]
    }

    optimize_parameter_pipeline(param_dict, "data\\benchmark_maps\\prepared_for_testing", 10)
