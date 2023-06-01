import numpy as np
import multiprocessing
import itertools
import pandas as pd
import os

from utils_preprocessing import *
from testing_and_visualize_map_preprocessing import *
from model_preprocessing import model_preprocessing


def evaluate_params(kernel_size=3, rho=1.0, theta=np.pi/180, threshold=30, minLineLength=30, maxLineGap=1, lengths_divisor=10,
                    eps=3.0, min_samples=15, show_img=False):
    """
    Rotate randomly the images and try to rotate the image back. The Loss is the difference between validated rotation angle and predicted rotation angle
    Arguments: parameter to test
    Return:
        Loss: difference between validated rotation angle and predicted rotation angle per image"""
    # create 12 randomly rotated images of the folder "data\benchmark_maps\prepared_for_testing"
    # rotated_imgs[0] -> filename, rotated_imgs[1] -> image, rotated_imgs[2] -> rotated angle
    rotated_imgs = create_test_imgs()
    # create empty lists to store the objects of map_preprocessing, the difference between validated rotation angle and predicted rotation angle and the count of detected lines
    preprocessed_imgs, diffs, lines = [], [], []
    # save the accumulation of differences between validated rotation angle and predicted rotation angle in loss
    loss = 0
    # loop through the test set of images
    for img in rotated_imgs:
        # preprocess the test images
        preprocessed = model_preprocessing(img[1], kernel_size=kernel_size, rho=rho, theta=theta, threshold=threshold,
                                           minLineLength=minLineLength, maxLineGap=maxLineGap, lengths_divisor=lengths_divisor, eps=eps, min_samples=min_samples)
        # get the rotated degree out of the instance variable
        rotated_degree = preprocessed.largest_cluster_of_the_orientations
        # calculate the difference between validated rotation angle and predicted rotation angle
        diff: float = round(abs(img[2] - rotated_degree), 1)
        # It doesn't matter whether the map is rotated to align the most parallel lines with the x-axis or the y-axis. Therefore, the maximum error is 45°
        if diff > 45.0:
            diff = round(90-diff, 1)
        # accumulate the differences
        loss = loss + diff

        if show_img:  # if we want to show the images, their scores and the count of detected lines, we have to save the informations
            preprocessed_imgs.append(img[1])
            preprocessed_imgs.append(preprocessed.rotated_image)
            line = len(preprocessed.angles_of_walls)
            diffs.append(str("original"))
            diffs.append("Fehler: " + str(diff) + " Linien erkannt: " + str(line))
    if show_img:
        show_all_imgs(preprocessed_imgs, diffs)
        print("Score aller Bilder: " + str(loss))
        print("Im durchschnitt ist ein Bild um " + str(loss/len(rotated_imgs)) + " verdreht")
    return loss/len(rotated_imgs)


def param_dict_to_combinations_in_df(param_dict):
    """
    Arguments:
        param_dict: a dictionary, which keys are the parameter, that contains a list of possible values
    Return:
        df: a dataframe, that store all combinations of all possible parameter combinations"""
    # Use itertools.product to get all possible combinations of values
    value_combinations = itertools.product(*param_dict.values())

    # Create a list of dictionaries where each dictionary contains one combination of values
    dict_list = [dict(zip(param_dict.keys(), values)) for values in value_combinations]

    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(dict_list)

    return df


def test_all_param_combinations(param_dict, evaluation_runs=10):
    """Takes the parameters to be tested with the corresponding values from the dictionaries and calculates the error deviation of the rotation.
    Arguments:
        param_dict: contains the names of the parameters as keys and the values to be tested as lists in the values
        evaluation_runs: In the evaluate_params function, the maps are randomly rotated and we predict this random rotation angle. Every run a different loss comes out.
                         To reduce randomness, we run this calculation more often
    Returns:
        df: a dataframe, that store all combinations of all possible parameter combinations with their corrospondending loss"""
    df = param_dict_to_combinations_in_df(param_dict)
    # create a new column "loss" and write the value 99999999 in each row.
    df["loss"] = 99999999
    # iterate through the dataframe and run every
    for index, row in df.iterrows():
        # set the loss on zero every iteration
        loss = 0.0
        # run the evaluation as often as described in the parameter evaluation_runs
        for i in range(evaluation_runs):
            loss += evaluate_params(kernel_size=int(row["kernel_size"]),
                                    rho=row["rho"], theta=row["theta"], threshold=int(row["threshold"]), minLineLength=row["minLineLength"], maxLineGap=row["maxLineGap"], eps=row["eps"], min_samples=int(row["min_samples"]), show_img=False)
            print(loss)
        # calculate the average loss
        loss = round(loss/evaluation_runs, 2)
        loss = str(loss) + str(" Parameter: Kernel ") + str(row["kernel_size"]) + str(" rho ") + str(row["rho"])
        # write the calculated loss in the loss column and the current row
        df.at[index, 'loss'] = loss
        """loss = process_row(row, evaluation_runs)
        df.at[index, 'loss'] = loss"""

    version = 1
    # before the csv is saved, the current version number must be found out
    if os.path.isfile('parameter_tuning\\tested_parameters_version_1.csv'):
        # if it does, determine the next version number by counting the number of existing versions
        while os.path.isfile(f'parameter_tuning\\tested_parameters_version_{version}.csv'):
            version += 1

    # write the DataFrame to a CSV file
    df.to_csv(f'parameter_tuning\\tested_parameters_version_{version}.csv', index=False)
    return df


def process_row(row, evaluation_runs):
    """
    _summary_

    Parameters
    ----------
    row : _type_
        _description_
    evaluation_runs : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    # set the loss on zero every iteration
    loss = 0.0
    # run the evaluation as often as described in the parameter evaluation_runs
    for i in range(evaluation_runs):
        loss += evaluate_params(kernel_size=int(row["kernel_size"]),
                                rho=row["rho"], theta=row["theta"], threshold=int(row["threshold"]), minLineLength=row["minLineLength"], maxLineGap=row["maxLineGap"], eps=row["eps"], min_samples=int(row["min_samples"]), show_img=False)
        print(loss)
    # calculate the average loss
    loss = round(loss/evaluation_runs, 2)
    # Test
    loss = str(loss) + str(" Parameter ") + str(row["kernel_size"]) + str(row["rho"]) + str(row["theta"]) + str(
        row["threshold"]) + str(row["minLineLength"]) + str(row["maxLineGap"]) + str(row["eps"]) + str(row["min_samples"])
    # return the calculated loss
    return loss


def parameter_tuning():
    # Defining the parameter ranges for the search
    param_dist = {
        # bigger outlines -> more line detections -> bigger line -> more inaccurate orientation
        "kernel_size": 3,
        "rho": 0.8,
        "theta": np.pi/360,
        "threshold": 50,
        "minLineLength": 60,
        "maxLineGap": 2,
        "lengths_divisor": 20,
        "eps": [0.5, 1, 3],
        "min_samples": 10
    }

    # Initialize variables for storing the best parameters and score
    best_param = ""
    best_score = float('-inf')
    progress = 0
    param_combination = {}

    threshold = param_dist["threshold"]
    lengths_divisor = param_dist["lengths_divisor"]

    # Loop through all parameter combinations
    with open("parameter_tuning_test.txt", "w") as file:
        with multiprocessing.Pool() as pool:
            for i, kernel in enumerate(param_dist["kernel_size"]):
                for u, rho in enumerate(param_dist["rho"]):
                    for theta in param_dist["theta"]:
                        # for threshold in param_dist["threshold"]:
                        for minLineLength in param_dist["minLineLength"]:
                            for maxLineGap in param_dist["maxLineGap"]:
                                # for lengths_divisor in param_dist["lengths_divisor"]:
                                for eps in param_dist["eps"]:
                                    for min_samples in param_dist["min_samples"]:
                                        # Test the current set of parameters and get the score
                                        score = pool.apply_async(evaluate_params, args=(
                                            kernel, rho, theta, threshold, minLineLength, maxLineGap, lengths_divisor, eps, min_samples)).get()
                                        param_combination["kernel_size= " + str(kernel) + " ,rho= " + str(rho) + " ,theta= " + str(theta) + " ,threshold= " + str(
                                            threshold) + " ,minLineLength= " + str(minLineLength) + " ,maxLineGap= " + str(maxLineGap) + " ,lengths_divisor= " + str(lengths_divisor) + " ,eps= " + str(eps) + " ,min_samples= " + str(min_samples)] = score
                                        file.write("kernel_size= " + str(kernel) + " ,rho= " + str(rho) + " ,theta= " + str(theta) + " ,threshold= " + str(
                                            threshold) + " ,minLineLength= " + str(minLineLength) + " ,maxLineGap= " + str(maxLineGap) + " ,lengths_divisor= " + str(lengths_divisor) + " ,eps: " + str(eps) + " ,min_samples= " + str(min_samples) + "--> score= " + str(score) + "\n")
                                        # Update the best parameters and score if the current score is better
                                        print("aktueller param: " + "kernel_size= " + str(kernel) + " ,rho= " + str(rho) + " ,theta= " + str(theta) + " ,threshold= " + str(
                                            threshold) + " ,minLineLength= " + str(minLineLength) + " ,maxLineGap= " + str(maxLineGap) + " ,lengths_divisor= " + str(lengths_divisor) + " ,eps: " + str(eps) + " ,min_samples= " + str(min_samples))
                                        print("aktueller score: " + str(score))
                    progress = progress + ((u+1)/len(param_dist["kernel_size"]))/len(param_dist["kernel_size"])
                    print("Progres: " + str(progress))

    # Output of the best model and parameters
    best_param, best_score = min(param_combination.items(), key=lambda x: x[1])

    print("Best parameters: " + best_param)
    print("Best score: " + str(best_score))

    # Save the output in a text file
    # Open a new text file in write mode
    with open("parameter_tuning_test.txt", "w") as file:
        # Write the values of param and score to the file
        file.write(f"Parameter: {best_param}\nScore: {best_score}")


"""def test_all_param_combinations(param_dict, evaluation_runs=10):
    '''Takes the parameters to be tested with the corresponding values from the dictionaries and calculates the error deviation of the rotation.
    Arguments:
        param_dict: contains the names of the parameters as keys and the values to be tested as lists in the values
        evaluation_runs: In the evaluate_params function, the maps are randomly rotated and we predict this random rotation angle. Every run a different loss comes out.
                         To reduce randomness, we run this calculation more often
    Returns:
        df: a dataframe, that store all combinations of all possible parameter combinations with their corrospondending loss'''

    df = param_dict_to_combinations_in_df(param_dict)
    # create a new column "loss" and write the value 99999999 in each row.
    df["loss"] = 99999999
    df["test"] = "Loss 2 richtige parameter"

    # create a ThreadPoolExecutor with 4 threads (you can adjust this as needed)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    # submit a job for each row of the DataFrame
    jobs = []
    for index, row in df.iterrows():
        job = executor.submit(process_row, row, evaluation_runs)
        jobs.append(job)

    # wait for all jobs to finish and update the DataFrame with the results
    for index, job in enumerate(concurrent.futures.as_completed(jobs)):
        loss = job.result()
        df.at[index, 'loss'] = loss

    # before the csv is saved, the current version number must be found out
    if os.path.isfile('parameter_tuning\\tested_parameters_version_1.csv'):
        # if it does, determine the next version number by counting the number of existing versions
        version = 1
        while os.path.isfile(f'parameter_tuning\\tested_parameters_version_{version}.csv'):
            version += 1

    # write the DataFrame to a CSV file
    df.to_csv(f'parameter_tuning\\tested_parameters_version_{version}.csv', index=False)
    return df"""

# define a function to process each row of the DataFrame


"""# test all parameters
    param_combination = {}
    for i, kernel in enumerate(param_dist["kernel_size"]):
        for u, rho in enumerate(param_dist["rho"]):
            for theta in param_dist["theta"]:
                for threshold in param_dist["threshold"]:
                    for minLineLength in param_dist["minLineLength"]:
                        for maxLineGap in param_dist["maxLineGap"]:
                            score = evaluate_params(kernel_size=kernel, rho=rho, theta=theta, threshold=threshold,
                                                  minLineLength=minLineLength, maxLineGap=maxLineGap, show_img=False)
                            param_combination["Kernel: " + str(kernel) + " rho: " + str(rho) + " theta: " + str(theta) + " Threshold: " + str(
                                threshold) + " minLineLength: " + str(minLineLength) + " maxLineGap: " + str(maxLineGap)] = score
                            print("aktueller score: " + str(score))
            progress = (i+1)/len(param_dist["kernel_size"]) + \
                ((u+1)/len(param_dist["kernel_size"])*(i+1)/len(param_dist["kernel_size"]))
            print("Progres: " + str((i+1)/len(param_dist["kernel_size"])))

    # Output of the best model and parameters
    param = str(min(param_combination, key=param_combination.get))
    score = param_combination.get(param)
    print("Best parameters: " + param)
    print("Best score:" + str(score))
    # Save the output in a text file
    # Open a new text file in write mode
    with open("parameter_tuning_test.txt", "w") as file:
        # Write the values of param and score to the file
        file.write(f"Parameter: {param}\nScore: {score}")"""


if __name__ == '__main__':
    ...
