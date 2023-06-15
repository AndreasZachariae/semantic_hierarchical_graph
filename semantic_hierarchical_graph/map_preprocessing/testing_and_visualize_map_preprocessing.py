# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Local/application specific imports
from utils_preprocessing import *


def show_all_imgs_of_instance(model_preprocessing_object):
    """
    Display all 2D numpy.ndarray images stored as attributes of the model_preprocessing_object in a grid.

    This method iterates through all attributes of the provided object, checks if the attribute value is 
    a 2D numpy.ndarray (which we assume corresponds to an image), and then displays these images in a grid format. 
    If there are more images than can fit in a single row, multiple rows are used. The images are displayed in grayscale. 

    Parameters
    ----------
    model_preprocessing_object : object
        The object whose attributes will be checked for 2D numpy.ndarray and displayed. This object is expected 
        to have several 2D numpy.ndarray attributes corresponding to images. 

    Returns
    -------
    None

    Displays
    --------
    grid : matplotlib.pyplot.subplots
        Grid of images displayed in grayscale. Each subplot's title corresponds to the name of the attribute 
        from which the displayed 2D numpy.ndarray was obtained. 

    Note
    ----
    The function does not return any value but directly visualizes the images using matplotlib."""
    # Create empty lists to store the images and their corresponding names
    imgs = []
    img_names = []
    # Loop through all attributes of the object
    for name, value in vars(model_preprocessing_object).items():
        # Check if the attribute is not a list and is a 2D array
        if isinstance(value, np.ndarray) and len(value.shape) == 2:
            # If the attribute satisfies the above condition, append it to the lists
            if len(value.shape) == 2:
                imgs.append(value)
                img_names.append(name)
    # Calculate the number of rows and columns required to display all the images in a grid
    num_rows = 2
    num_cols = len(imgs) // num_rows + 1
    # Create a Matplotlib figure object with the specified number of rows and columns
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(18, 9))
    # Loop through the images and their names and display each image in the grid
    for i, (img, name) in enumerate(zip(imgs, img_names)):
        # Calculate the row and column where the image should be displayed
        row = i // num_cols
        col = i % num_cols
        # Use Matplotlib's imshow() function to display the image in grayscale
        axs[row][col].imshow(img, cmap='gray')
        # Set the title of the image to its corresponding name
        axs[row][col].set_title(name)
    # Display the grid of images
    plt.show()


def plot_cluster(X, labels, db, largest_cluster_centroid, axs):
    """This function is for visualize the results of DBSCAN clustering.
    Set the parameter "bool_plot_clustering" in the function "find_largest_cluster_of_the_orientations"
    to "true" to see the results"""
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    # plot the results
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]  # type: ignore
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
        class_member_mask = labels == k
        xy = X[class_member_mask & core_samples_mask]
        axs.plot(xy[:], xy[:], "o", markerfacecolor=tuple(col), markeredgecolor="k", markersize=14,)
        xy = X[class_member_mask & ~core_samples_mask]
        axs.plot(xy[:], xy[:], "o", markerfacecolor=tuple(col), markeredgecolor="k", markersize=6,)
    if largest_cluster_centroid is not None:
        axs.plot(largest_cluster_centroid, largest_cluster_centroid, 's',
                 markerfacecolor='yellow', markersize=12, markeredgecolor='black')
    axs.set_title(f"Estimated number of clusters: {n_clusters_}")


def create_test_imgs(filenames_imgs=None):
    """
    Read in all images from a specified file, apply a random rotation, and return the processed images.

    This function reads all images from a directory specified in the function call to 
    'read_in_all_images_from_file'. Each image is then rotated by a random angle using 
    the 'randomly_rotate' function. 

    Returns
    -------
    rotated_imgs : list
        A list where each element is a tuple containing the filename, the processed image, 
        and the angle by which the image has been rotated.

    Note
    ----
    The directory from which images are read is hardcoded into the function and may need to be adjusted depending on the file structure.
    """
    if filenames_imgs is None:
        filenames_imgs = read_in_all_images_from_file(
            "data\\benchmark_maps\\prepared_for_testing")
    rotated_imgs = randomly_rotate(filenames_imgs)
    return rotated_imgs  # rotated_imgs[0] -> filename, rotated_imgs[1] -> image, rotated_imgs[2] -> rotated angle


def show_all_imgs(imgs, titles=None):
    """
    Display all images from a given list, with optional titles.

    This function displays all images from a list passed as the 'imgs' argument. Images are 
    displayed in a grid using matplotlib's imshow() function. If a list of titles is provided, 
    these titles are used to label the corresponding images.

    Parameters
    ----------
    imgs : list
        A list of images. Each image should be a 2D array (grayscale) or 3D array (color).

    titles : list of str, optional
        A list of titles corresponding to the images. If provided, each image will be labeled 
        with its corresponding title. The number of titles should match the number of images. 
        Default is None, in which case images are not labeled.

    Example
    -------
    >>> img1 = np.array([[255, 255], [0, 0]], dtype=np.uint8)
    >>> img2 = np.array([[0, 0], [255, 255]], dtype=np.uint8)
    >>> show_all_imgs([img1, img2], ['Image 1', 'Image 2'])

    Note
    ----
    The images in the list should be numpy arrays, as expected by matplotlib's imshow() function.
    """
    plt.figure(figsize=(29, 29))
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.05, right=0.95, hspace=0.5, wspace=0.4)
    for i, img in enumerate(imgs):
        plt.subplot(4, int(len(imgs)/4)+1, i+1)
        plt.imshow(img, cmap='gray')
        if titles is not None:
            plt.title(titles[i])
    plt.show()
