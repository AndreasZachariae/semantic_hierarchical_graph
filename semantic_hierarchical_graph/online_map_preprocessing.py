import cv2
from semantic_hierarchical_graph import segmentation
from matplotlib import pyplot as plt
import os
import random
from PIL import Image
"""
Description:
For each image, it applies black and white thresholding to create binary images and then combines the thresholded images. 
So the online resources images have only a pixel value of 0 (black), 205 (gray) and 254(white)

Functions:

    read_in_all_images_from_file(path)
        This function takes the path of a directory containing images as input.
        It reads in all the image files with extensions .jpg, .jpeg and .png from the specified directory.
        The function returns a list of tuples containing the filename and the image content.

    thresh_black_white_gray(path)
        This function takes the path of a directory containing images as input.
        It calls the read_in_all_images_from_file() function to get a list of images.
        For each image, it applies black and white thresholding to create binary images.
        The user is prompted to select threshold values for black and white.
        It then combines the thresholded images and displays them for the user to verify.
        The user is given the option to save the resulting image or adjust the threshold values.
        The function saves the resulting image to a directory called "prepared_for_testing" if the user selects to save it."""


def read_in_all_images_from_file(path):
    image_files = []
    # Loop through all files in the directory
    for file in os.listdir(path):
        # Check if the file is an image
        if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
            # Read in the image and store its filename and content in a tuple
            image = cv2.imread(os.path.join(path, file))
            image_files.append((file, image))
    return image_files


def thresh_black_white_gray(path):
    """This function show you the image and you can input a threshold value for black, white and gray as long as you are happy with the result"""
    imgs = read_in_all_images_from_file(path)
    for img in imgs:
        happy_with_img = False
        while not happy_with_img:
            print("select thresh value for black")
            segmentation.show_imgs(img[1])
            black = input()
            ret, thresh_img_black = cv2.threshold(img[1], float(black), 205, cv2.THRESH_BINARY)
            print("select thresh value for white")
            white = input()
            ret, thresh_img_white = cv2.threshold(img[1], float(white), 49, cv2.THRESH_BINARY)
            result = cv2.add(thresh_img_black, thresh_img_white)
            cv2.putText(thresh_img_black, "black values",
                        (int(thresh_img_black.shape[1]/2)-60, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
            cv2.putText(thresh_img_white, "white values",
                        (int(thresh_img_white.shape[1]/2)-60, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
            segmentation.show_imgs(thresh_img_black, thresh_img_white)
            print("If you're happy with the image, press 'y' to save it.")
            print("If you don't want to save the image, press 'n'.")
            print("If you want to adjust the black and white threshold values, press any other key.")
            segmentation.show_imgs(result)
            response = input()
            if response == "y":
                cv2.imwrite("data\\benchmark_maps\\prepared_for_testing\\" + str(img[0]), result)
                happy_with_img = True
            if response == "n":
                break


def rotate(img, angle_degrees):
    # Determine the center of the image
    height, width = img.shape[:2]  # image shape has 3 dimensions
    # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
    image_center = (width/2, height/2)

    # Define the rotation matrix
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle_degrees, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(img, rotation_mat, (bound_w, bound_h), borderValue=[
                                 205, 205, 205], flags=cv2.INTER_NEAREST)

    return rotated_mat


def randomly_rotate(imgs):
    rotated_imgs = []
    for img in imgs:
        random_number = random.uniform(0, 90)
        # gray_img = cv2.cvtColor(img[1], cv2.COLOR_BGR2GRAY)
        # Determine the dimensions of the image
        height, width = img[1].shape[:2]
        # Set the amount to crop from each edge
        crop_amount = 2
        # Crop the image by slicing the array
        # some images have a thin frame, with croping it will be removed
        cropped_img = img[1][crop_amount:height-crop_amount, crop_amount:width-crop_amount]
        rotated_img = rotate(cropped_img, random_number)
        rotated_imgs.append((img[0], rotated_img, random_number))
    return rotated_imgs


def correct_the_degree_manually(imgs):
    for img in imgs:
        unhappy = True
        image = img[1]
        while unhappy:
            print("press the degree value: ")
            segmentation.show_imgs(image)
            degree = input()
            image = rotate(image, float(degree))
            print("if you are happy and want to save, than press: 'y'")
            print("if you want to try again, than press: 'r")
            print("print 'q' to quit")
            segmentation.show_imgs(image)
            response = input()
            if response == 'y':
                unhappy = False
                cv2.imwrite("data\\benchmark_maps\\prepared_for_testing\\" + str(img[0]), image)
            if response == 'q':
                unhappy = False


def scale_img_to_double_size():
    # Load the image
    # img = cv2.imread("data\\benchmark_maps\\prepared_for_testing\\seattle_r_scaled.png")
    img = Image.open(
        "data\\benchmark_maps\\prepared_for_testing\\intel_research_lab.png").convert('L')
    segmentation.show_imgs(img)
    # Get the original size
    # height, width = img.size[:2]
    width, height = img.size
    # print(height)
    # Create a new image with double size
    new_img = Image.new("L", (width*2, height*2))
    # Scale the image using nearest neighbor interpolation
    for x in range(width*2):
        for y in range(height*2):
            pixel = img.getpixel((x/2, y/2))
            new_img.putpixel((x, y), pixel)
    # Save the scaled image
    new_img.save("data\\benchmark_maps\\prepared_for_testing\\intel_research_lab_scaled.png")


if __name__ == '__main__':
    # thresh_black_white_gray("data\\benchmark_maps\\online_ressources\\new")
    # correct_the_degree_manually(read_in_all_images_from_file("data\\benchmark_maps\\online_ressources\\new"))
    scale_img_to_double_size()
