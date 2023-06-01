import cv2
from semantic_hierarchical_graph import segmentation
from matplotlib import pyplot as plt
import os
import random
from PIL import Image
from utils_preprocessing import *


class edit_maps_from_online_ressources:
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

    @ staticmethod
    @ staticmethod
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

    """@ staticmethod
    def rotate(img, angle_degrees):
        
        Rotates the given image around the given angle

        Parameters
        ----------
        img : _type_
            Image to rotate
        angle_degrees : Float
            Angle to rotate in degree

        Returns
        -------
        _type_
            rotated image
        
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

        return rotated_mat"""

    @ staticmethod
    @ staticmethod
    def correct_the_degree_manually(imgs):
        """
        Function to create a groundtruth of the correct Angle. With this function you can correct the degree manually by user input and save the image.

        Parameters
        ----------
        imgs : _type_
            false rotated image
        """
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


if __name__ == '__main__':
    ...
