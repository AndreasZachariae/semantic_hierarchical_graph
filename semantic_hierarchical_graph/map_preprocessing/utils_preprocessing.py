# Standard library imports
import cProfile
import os
import random
import yaml

# Third-party imports
import cv2
import numpy as np


def testing_speed():
    """shows me the runtime of the functions"""
    cProfile.run('map_preprocessing(img1)')


def threshold(img, thresh=100, maxval=255):
    ret, thresh_img = cv2.threshold(img, thresh, maxval, cv2.THRESH_BINARY)
    return thresh_img


def save_img_as_png(img, filename="saved_image"):
    """
    takes an image and save it as png with the called title

    Parameters
    ----------
    img : _type_
        image to convert and save
    filename : str, optional
        filename of the image, by default "saved_image"
    """
    # Convert image to PNG format
    png_img = cv2.imencode('.png', img)[1]

    # Save PNG image to disk
    filename = filename + ".png"
    with open(filename, 'wb') as f:
        f.write(png_img)


def rotate(img, degree):
    """
    Rotates the image by a specified degree and crops it to the optimal size.

    This function first expands the original image to prevent corner information loss during rotation.
    Then, it applies a rotation transformation and crops the resulting image to a bounding box that
    contains all non-zero pixels. This effectively removes unnecessary padding from the rotated image.

    Parameters
    ----------
    degree : float
        The degree of clockwise rotation to apply to the image.

    Returns
    -------
    numpy.ndarray
        A processed image which is first rotated by the specified degree and then cropped
        to minimize the size while preserving all relevant content. The returned image
        is in grayscale format with the same data type as the original.
    """
    # Get the dimensions of the image
    height, width = img.shape[:2]
    # expand the image with a padding of the size of the image's diagonal length. This length is choosed randomly. It just have to be large enough
    diagonal = int((height**2 + width**2)**0.5)
    padded_img = cv2.copyMakeBorder(img, diagonal, diagonal, diagonal,
                                    diagonal, cv2.BORDER_CONSTANT, value=[205, 205, 205])
    # calculate the RotationMatrix of the expanded/ padded image
    new_height, new_width = padded_img.shape[:2]
    center = (new_width / 2, new_height / 2)
    M = cv2.getRotationMatrix2D(center, degree, 1.0)
    # rotate the padded image
    converted_img = cv2.convertScaleAbs(padded_img)
    rotated_img = cv2.warpAffine(converted_img, M, (new_width, new_height),
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=205, flags=cv2.INTER_NEAREST)
    # Now the padded image is larger than it should be. In the next steps we crop the image to a nice size
    # Threshold to create a binary image
    thresh = cv2.threshold(rotated_img, 250, 255, cv2.THRESH_BINARY)[1]
    # Get bounding box
    x, y, w, h = find_bounding_box(thresh)
    # Crop bbox for perfect image size
    crop_x1, crop_y1 = max(0, int(x-20)), max(0, int(y-20))
    crop_x2, crop_y2 = min(rotated_img.shape[1], x+w+20), min(rotated_img.shape[0], y+h+20)

    return rotated_img[crop_y1:crop_y2, crop_x1:crop_x2]


def randomly_rotate(filenames_imgs):
    """
    Takes a list of images and rotate every image around a random angle and save the random angle

    Parameters
    ----------
    imgs : _type_
        List, that contains a list of filename and image to rotate

    Returns
    -------
    List
        [0] Filename of the rotated image
        [1] rotated image
        [2] rotated angle

    """
    rotated_imgs = []
    for filenames_img in filenames_imgs:
        random_number = random.uniform(0, 90)
        # gray_img = cv2.cvtColor(img[1], cv2.COLOR_BGR2GRAY)
        # Determine the dimensions of the image
        height, width = filenames_img[1].shape[:2]
        # Set the amount to crop from each edge
        crop_amount = 2
        # Crop the image by slicing the array
        # some images have a thin frame, with croping it will be removed
        cropped_img = filenames_img[1][crop_amount:height-crop_amount, crop_amount:width-crop_amount]
        rotated_img = rotate(cropped_img, random_number)
        rotated_imgs.append((filenames_img[0], rotated_img, random_number))
    return rotated_imgs


def find_bounding_box(threshed_image):
    """
    Determines the coordinates of a bounding box that contains all non-zero pixels in a binary image.

    This function finds all non-zero pixels in the given image and returns their minimum and maximum x and y coordinates,
    which together form the bounding box.

    Parameters:
    threshed_image (numpy.ndarray): A binary image, where non-zero pixels are considered as 'object' pixels.

    Returns:
    tuple: A tuple containing four elements in the following order:
        min_x (int): The minimum x-coordinate of non-zero pixels.
        max_x (int): The maximum x-coordinate of non-zero pixels.
        min_y (int): The minimum y-coordinate of non-zero pixels.
        max_y (int): The maximum y-coordinate of non-zero pixels.
    """
    # Find the coordinates of all nonzero pixels in the image
    nonzero_pixels = np.nonzero(threshed_image)
    # Get the minimum and maximum x and y coordinates of the nonzero pixels
    min_x = np.min(nonzero_pixels[1])
    max_x = np.max(nonzero_pixels[1])
    min_y = np.min(nonzero_pixels[0])
    max_y = np.max(nonzero_pixels[0])

    # Return the bounding box coordinates as a tuple
    return (min_x, min_y, max_x - min_x, max_y - min_y)


def load_yaml(filepath):
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)


def read_in_all_images_from_file(path):
    """
    This function takes the path of a directory containing images as input.
        It reads in all the image files with extensions .jpg, .jpeg and .png from the specified directory.

    Parameters
    ----------
    path : String
        location of the file

    Returns
    -------
    List
        List of all images and filenames in this folder
    """
    image_files = []
    # Loop through all files in the directory
    for file in os.listdir(path):
        # Check if the file is an image
        if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
            # Read in the image and store its filename and content in a tuple
            image = cv2.imread(os.path.join(path, file))
            image_files.append((file, image))
    return image_files


if __name__ == '__main__':
    ...
