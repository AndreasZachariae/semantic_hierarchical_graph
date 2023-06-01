import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.cluster import DBSCAN

from semantic_hierarchical_graph import segmentation
from semantic_hierarchical_graph.types.parameter import Parameter
from semantic_hierarchical_graph.types.vector import Vector
from testing_and_visualize_map_preprocessing import *
from utils_preprocessing import *


class model_preprocessing:
    """
    The class takes an input image and performs a series of operations to produce a preprocessed image.

    The class begins by converting the input image into grayscale format. It then applies a binary threshold to recognize black pixels, 
    which represent barriers in the image. These barriers are then dilated to create continuous lines. Following this, incorrectly measured 
    values within the image are removed.

    The class also finds the orientation of walls in the image and determines the lengths of these walls. 
    The angles and lengths are used to identify clusters of walls, with the orientation of the largest cluster being used to rotate the image. 
    The end result is a processed image that is oriented horizontally according to the X-axis.

    Attributes
    ----------
    gray_image : ndarray
        A grayscale version of the input image.
    black_pixels_image : ndarray
        A binary image where black pixels from the original image are converted to white.
    white_pixels_image : ndarray
        A binary image of open space.
    dilated_outline : ndarray
        An image where the recognized barriers are dilated to create continuous lines.
    removed_incorrect_measured_values : ndarray
        An image where incorrectly measured values are removed.
    angles_of_walls : list
        A list of the orientations of walls in the image.
    lengths_of_walls : list
        A list of the lengths of the walls in the image.
    image_with_orientations : ndarray
        An image with the orientations of the walls marked on it.
    largest_cluster_of_the_orientations : float
        The angle of the largest cluster of orientations.
    rotated_image : ndarray
        The final, preprocessed image which is rotated according to the largest cluster of orientations.

    Methods
    -------
    __init__(self, img, kernel_size=5, rho=1.0, theta=np.pi/180, threshold=30, minLineLength=35, maxLineGap=2, lengths_divisor=10, eps=1.0, min_samples=10):
        Initializes the model_preprocessing class with the specified parameters and applies various preprocessing methods to the input image.
    threshold(img, thresh=100, maxval=255):
        Applies a threshold to the input image.
    thresh_black_and_return_as_white(img):
        Returns a binary image with original black pixels converted to white.
    dilate_outline(img, kernel_size=3):
        Dilates the recognized barriers in the image to create continuous lines.
    remove_small_areas_and_write_as_grayscale(img, grayscale_value=255, min_area=400):
        Removes small areas from the image and writes the result as a grayscale image.
    remove_incorrect_measured_values(self, kernel_size):
        Removes incorrectly measured values from the image.
    erode_corrected_outline(self, img, kernel_size=3):
        Erodes the corrected outline of the image.
    get_orientation_of_walls(self, rho=1.0, theta=np.pi/180, threshold=30, minLineLength=30, maxLineGap=1):
        Retrieves the orientation of walls in the image.
    find_largest_cluster_of_the_orientations(self, lengths_divisor=10, eps=3.0, min_samples=10, bool_plot_clustering=False):
        Identifies the largest cluster of orientations in the image.
    """

    def __init__(self, img, **kwargs):
        # Provide default values and update them with any provided kwargs
        self.params = {
            'kernel_size': 5,
            'rho': 1.0,
            'theta': np.pi/180,
            'threshold': 30,
            'minLineLength': 35,
            'maxLineGap': 2,
            'lengths_divisor': 10,
            'eps': 1.0,
            'min_samples': 10
        }
        self.params.update(kwargs)
        # Convert the input image to grayscale
        self.gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply a threshold to the binary image to recognize black pixels(i.e., the barriers) and return an image with white pixels at the locations of the recognized barriers
        self.black_pixels_image = self.thresh_black_and_return_as_white(self.gray_image)
        # Create binary image of open space
        self.white_pixels_image = threshold(self.gray_image, thresh=250, maxval=255)
        # Dilate the outline of the recognized barriers to get a continious line
        self.dilated_outline = self.dilate_outline(self.black_pixels_image, self.params['kernel_size'])
        # Remove incorrectly measured values from the image
        self.removed_incorrect_measured_values = self.remove_incorrect_measured_values(self.params['kernel_size'])
        # Get the orientation of walls in the image and the lengths of the walls
        # as well as an image with wall orientations marked on it
        self.angles_of_walls, self.lengths_of_walls, self.image_with_orientations = self.get_orientation_of_walls(
            self.params['rho'], self.params['theta'], self.params['threshold'], self.params['minLineLength'], self.params['maxLineGap'])
        # Save largest cluster of the orientations
        self.largest_cluster_of_the_orientations = self.find_largest_cluster_of_the_orientations(self.params['lengths_divisor'], self.params['eps'], self.params['min_samples'],
                                                                                                 bool_plot_clustering=False)
        # Rotate the image based on the orientation of the walls
        self.rotated_image = rotate(self.removed_incorrect_measured_values,
                                    int(-(self.largest_cluster_of_the_orientations)))

    """def threshold(img, thresh=100, maxval=255):
        ret, thresh_img = cv2.threshold(img, thresh, maxval, cv2.THRESH_BINARY)
        return thresh_img"""

    @ staticmethod
    def thresh_black_and_return_as_white(img):
        """
        Converts black pixels in a grayscale image to white and inverses the color.

        This function applies a binary threshold to the input image, identifying black pixels
        which typically represent barriers or obstacles. It then creates an inverse image 
        where the originally black pixels are turned to white.

        Parameters
        ----------
        img : numpy.ndarray
            A grayscale input image to be processed. 

        Returns
        -------
        numpy.ndarray
            An inversed binary image where the original black pixels are turned white, 
            and the remaining pixels are turned black. The returned image maintains the same 
            dimensions as the input image.

        Note
        ----
        The input image should be a grayscale image for appropriate processing.
        """
        # thresh all black pixels to recognise the barriers
        ret, thresh_black_pixels = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
        # Get the height and width of the thresholded image
        height, width = thresh_black_pixels.shape
        # Create a new 2D numpy array of size height x width with all elements set to 255
        mat_255 = np.full((height, width), 255, dtype=np.uint8)
        # Subtract the thresholded image from the new array to invert black and white pixels
        black_dots_inversed = cv2.subtract(mat_255, thresh_black_pixels)
        return black_dots_inversed

    @ staticmethod
    def dilate_outline(img, kernel_size=3):
        """
        Dilates the outlines of barriers within a given image to ensure continuous lines.

        In the context of this function, black pixels represent recorded barriers or obstacles. Due to imprecise recording, 
        minor gaps might be present in these barriers. This function employs dilation to fill these gaps, resulting in continuous lines. 
        Ensuring continuous lines is critical for the operation of 'cv2.connectedComponents' later in the pipeline.

        Parameters:
        img (numpy.ndarray): The input image with black pixels representing barriers or obstacles.
        kernel_size (int, optional): The size of the dilation kernel. Default is 3.

        Returns:
        numpy.ndarray: An image with the black pixels dilated, ensuring continuous barrier lines.
        """
        # expand/ dialte the black pixels to get continous lines
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_blackdots = cv2.dilate(img, kernel, iterations=1)
        return dilated_blackdots

    @ staticmethod
    def remove_small_areas_and_write_as_grayscale(img, grayscale_value=255, min_area=400):
        """
        Processes the given image to remove small areas or islands and converts the filtered image into grayscale.

        This function works by finding all connected components (or islands) in the input image. It then loops through each connected 
        component and if its area is less than the defined 'min_area', it removes it by setting the label to zero.
        The filtered binary image is then converted into a grayscale image by multiplying with the provided 'grayscale_value'.

        Parameters:
        img (numpy.ndarray): The input image
        grayscale_value (int, optional): The value to be used for the grayscale image. Default is 255, which represents white.
        min_area (int, optional): The minimum area threshold, areas smaller than this value will be removed. Default is 400.

        Returns:
        numpy.ndarray: an image with dilated black pixels
        """
        # find the connected components/ islands in the image
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
        # Loop through the connected components and filter out the ones with a small area
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                labels[labels == i] = 0
        # Convert the label image back to a binary image
        binary_filtered = (labels > 0).astype('uint8') * grayscale_value
        ret, thresh = cv2.threshold(binary_filtered, 1, grayscale_value, cv2.THRESH_BINARY)
        return thresh

    def remove_incorrect_measured_values(self, kernel_size):
        """
        Removes incorrect measurements from the original map, specifically the long thin lines that could occur when a 
        laser passes through a window or any other transparent obstacle.

        This function primarily deals with filtering out noise introduced by the Lidar when it fails to detect 
        an object. The steps involved include subtraction of the dilated outline from the white pixels image to create 
        a gap, then removal of small areas or noise from the map and the creation of a single-color image. 

        After cleaning up the measured values, the function applies a threshold to separate free space, unknown area, 
        and obstacles by assigning unique grayscale values. Lastly, it uses erosion to correct the outline of the final image.

        Parameters:
        kernel_size (int): The size of the kernel used for the erosion operation.

        Returns:
        numpy.ndarray: The corrected map image free from noise and incorrect measurements, where free space, unknown areas, and obstacles are distinctly visible with separate grayscale values.

        Note:
        This function uses `self.white_pixels_image` and `self.dilated_outline`, which are assumed to be images of white pixels and dilated outlines respectively, extracted from the original map.
        """
        # Subtract the dilated outline from the white pixels image to seperate the white pixels behind the windwos from the main map, so that a gap will be between
        image_with_thin_lines_as_islands = cv2.subtract(self.white_pixels_image, self.dilated_outline)

        # create image of the main map without the islands
        image_of_free_space = model_preprocessing.remove_small_areas_and_write_as_grayscale(
            image_with_thin_lines_as_islands, min_area=1400)

        # to remove the measured black dots behind a window, don't remove black areas from the dilated_outline image, because many obstacles within the room will be removed too
        # to remove the measured black dots behind a window, first merge the main
        image_of_free_space_and_dilated_outline_with_same_colour = cv2.add(image_of_free_space, self.dilated_outline)

        # Now, the map is one monochrome area and the measured dots outside the building are separated and can be removed.
        image_of_removed_incorrect_measured_boundary = model_preprocessing.remove_small_areas_and_write_as_grayscale(
            image_of_free_space_and_dilated_outline_with_same_colour, min_area=300)

        # To get the unmeasured area we need to invert the last image (free space and outline are 255 and everything else have to be unmeasured)
        ret, img_of_unknown_area_scaled_to_205 = cv2.threshold(
            image_of_removed_incorrect_measured_boundary, 0, 205, cv2.THRESH_BINARY_INV)

        # merge the free space (colour value = 255) and unknown area (colour value = 205), pixel that are neither free space nor unknown must be obstacles (colour value = 0). Obstacles are in both images "0", so after summation it is still "0"
        image_free_space_and_unknown_area_and_outline = cv2.add(image_of_free_space, img_of_unknown_area_scaled_to_205)

        # errode black outline
        corrected_image = self.erode_corrected_outline(image_free_space_and_unknown_area_and_outline, kernel_size)

        return corrected_image

    def erode_corrected_outline(self, img, kernel_size=3):
        """
        Performs erosion on the given image and corrects the eroded outline using the original grayscale image.

        This function first inverts the binary threshold of the image, creating a binary inverse of it, 
        and then applies morphological erosion operation to reduce noise or small structures in the image.
        It subsequently calculates the dilated pixels and restores their original grayscale values, 
        essentially correcting the eroded outline of the image.

        Parameters:
        img (numpy.ndarray): The input image, assumed to be grayscale.
        kernel_size (int, optional): The size of the kernel used for the erosion operation. Default is 3.

        Returns:
        numpy.ndarray: The image with the eroded outline corrected using the original grayscale image.

        Note:
        This function uses `self.gray_image`, which is assumed to be the original, unmodified grayscale image.
        """
        # thresh black outline
        ret, corrected_dilated_outline = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV)
        # Define the kernel for erosion
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        # Perform erosion on the thresholded image
        eroded_img = cv2.erode(corrected_dilated_outline, kernel, iterations=1)
        # recieve the dilated pixels
        dilated_pixels = cv2.subtract(corrected_dilated_outline, eroded_img)
        # restore the original values to the former dilated pixels
        # Use boolean indexing to replace pixels in dilated_pixels with corresponding pixels from gray_image
        img[dilated_pixels > 0] = self.gray_image[dilated_pixels > 0]
        return img

    def get_orientation_of_walls(self, rho=1.0, theta=np.pi/180, threshold=30, minLineLength=30, maxLineGap=1):
        """
        Detects the walls in the given image and calculates their orientations and lengths.

        This function detects the borders of the image by removing small areas and fitting straight lines to the outline using the Hough Line Transform. 
        For each detected wall, it calculates the orientation and length. If no lines are detected, it returns a default response.

        Parameters:
        rho (float, optional): Distance resolution of the accumulator in pixels. Default is 1.0.
        theta (float, optional): Angle resolution of the accumulator in radians. Default is np.pi/180.
        threshold (int, optional): Accumulator threshold parameter. Only those lines are returned that get enough votes (threshold). Default is 30.
        minLineLength (int, optional): Minimum line length. Line segments shorter than this are rejected. Default is 30.
        maxLineGap (int, optional): Maximum allowed gap between points on the same line to link them. Default is 1.

        Returns:
        list: A list of wall orientations in degrees.
        list: A list of wall lengths.
        numpy.ndarray: An image with the detected walls marked and their orientations written on it.
        """
        # Detect the border of the image
        border_img = model_preprocessing.remove_small_areas_and_write_as_grayscale(self.dilated_outline, min_area=200)
        # Fit straight lines to the outline
        lines = cv2.HoughLinesP(border_img, rho=rho, theta=theta, threshold=threshold,  # The Hough transform for detecting lines is often expressed in polar coordinates, with each point in the image space being transformed into a sinusoidal curve in the Hough space. The parameters rho and theta are used to represent the polar coordinates of the curves in the Hough space.
                                minLineLength=minLineLength, maxLineGap=maxLineGap)
        # Calculate the orientation and length of each wall
        angles = []
        lengths = []
        if lines is not None:
            for line in lines:
                # Get the coordinates of the endpoints of the line
                x1, y1, x2, y2 = line[0]
                # Create a vector from the endpoints
                point1 = (x1, self.dilated_outline.shape[0] - y1)
                point2 = (x2, self.dilated_outline.shape[0] - y2)
                vector = (Vector.from_two_points(point1, point2))
                # Draw the line on the image
                cv2.line(border_img, (x1, y1), (x2, y2), (100), 2)
                # Calculate the orientation in radians and convert it to degrees
                angle_radians = math.atan2(vector.y, vector.x)  # firstly y then x
                angle_degree = round(math.degrees(angle_radians), 2)
                if angle_degree < 0:
                    angle_degree = angle_degree+180
                # Write the degree of the wall on the image
                pos = ((x1 + x2)//2, (y1 + y2)//2)
                cv2.putText(border_img, str(angle_degree), pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
                # Add the angle and length to the lists
                angles.append(angle_degree)
                length = int(math.sqrt((vector.x)**2 + (vector.y)**2))
                lengths.append(length)
            # Return the angles, lengths of the walls and the image showing the detected walls
            return angles, lengths, border_img
        else:
            return [0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1], border_img  # if no lines detected -> rotation of 0°

    def find_largest_cluster_of_the_orientations(self, lengths_divisor=10, eps=3.0, min_samples=10, bool_plot_clustering=False):
        """
        Determines the dominant wall orientation in the given image by finding the largest cluster of similar angles.

        This function uses a two-pass approach to account for potential clusters split by the edges of the value range.
        In the first pass, it assigns a weight to each angle based on its length, with orthogonal walls treated as the same cluster.
        Then, it uses the DBSCAN clustering algorithm to group similar angles together.
        In the second pass, it repeats this process with the angle values shifted by 45 degrees.
        The dominant wall orientation is then determined by comparing the results of both runs based on the number of data points in each cluster.

        Parameters:
        lengths_divisor (int, optional): Determines the weight assigned to each angle based on its length. Default is 10.
        eps (float, optional): The maximum distance between two samples for one to be considered as in the neighborhood of the other in DBSCAN clustering. Default is 3.0.
        min_samples (int, optional): The number of samples in a neighborhood for a point to be considered as a core point in DBSCAN clustering. Default is 10.
        bool_plot_clustering (bool, optional): If True, plots the clustering results. Default is False.

        Returns:
        float: The dominant wall orientation in degrees. If no dominant orientation is found, returns 0.
        """
        # weight the angles of the walls with their length of the wall -> For example: a wall rotated by 30° with length 50, is written 5 times with 30° in the list.
        # A cluster could be divided by the edges of the value range, therefore two passes with 45° displacement
        angles_weighted_for_1st_run, angles_weighted_for_2nd_run = [], []
        for angle in range(len(self.angles_of_walls)):
            # to reduce computation time, step in the loop in the size of lengths divisor
            for lenght in range(0, self.lengths_of_walls[angle], lengths_divisor):
                angle_1 = self.angles_of_walls[angle] % 90  # orthogonal lines belong to the same cluster
                # a cluster could be divided by the edges of the value range, therefore two passes with 45° displacement
                angle_2 = (self.angles_of_walls[angle] + 45) % 90
                # e.g.: angles_weighted_for_1st_run = [39.0, 39.0, 39.0, 39.0, 39.0, ...]
                angles_weighted_for_1st_run.append(angle_1)
                angles_weighted_for_2nd_run.append(angle_2)

        # convert 1D data to 2D array(=line in the coordinate system), because DBSCAN works only with 2D
        X1 = np.array(angles_weighted_for_1st_run).reshape(-1, 1)  # X1 = [[39.] [39.] [39.] ... [21.] [21.] [21.]]
        X2 = np.array(angles_weighted_for_2nd_run).reshape(-1, 1)

        # perform DBSCAN clustering for both runs
        db1, db2 = DBSCAN(eps=eps, min_samples=min_samples), DBSCAN(eps=eps, min_samples=min_samples)
        db1.fit(X1)
        db2.fit(X2)
        labels1, labels2 = db1.labels_, db2.labels_  # labels1 = [0 0 0 ... 0 0 0]

        # count the number of data points in each cluster
        labels1_unique, counts1 = np.unique(db1.labels_, return_counts=True)
        labels2_unique, counts2 = np.unique(db2.labels_, return_counts=True)
        # find the cluster label with the largest number of data points
        largest_cluster_label1 = labels1_unique[np.argmax(counts1)]
        largest_cluster_label2 = labels2_unique[np.argmax(counts2)]
        # get the number of data points in the largest cluster
        largest_cluster_size1 = counts1[np.argmax(counts1)]
        largest_cluster_size2 = counts2[np.argmax(counts2)]
        # to find the largest cluster, compare the largest clusters of each of the two runs
        if largest_cluster_size1 >= largest_cluster_size2:
            # extract the data points belonging to the largest cluster
            largest_cluster_mask = db1.labels_ == largest_cluster_label1
            largest_cluster_points = X1[largest_cluster_mask]
            # compute the centroid of the largest cluster
            largest_cluster_centroid = np.mean(largest_cluster_points, axis=0)
        else:
            # extract the data points belonging to the largest cluster
            largest_cluster_mask = db2.labels_ == largest_cluster_label2
            largest_cluster_points = X2[largest_cluster_mask]
            # compute the centroid of the largest cluster
            largest_cluster_centroid = np.mean(largest_cluster_points, axis=0) - \
                45  # the angles were shifted by 45 degrees
        if bool_plot_clustering:
            fig, axs = plt.subplots(1, 2, figsize=(12, 4))
            plot_cluster(X1, labels1, db1, largest_cluster_centroid, axs[0])
            plot_cluster(X2, labels2, db2, largest_cluster_centroid, axs[1])
            plt.show()
        if largest_cluster_centroid is not None:
            return largest_cluster_centroid[0]
        else:
            return 0


if __name__ == '__main__':
    img = cv2.imread('data\\benchmark_maps\\prepared_for_testing\\ryu.png')
    test_all_intermediate_steps()

    # Create test images and store returned values
    rotated_imgs = create_test_imgs()

    # Preprocess each image and store returned model_preprocessing objects
    preprocessed_imgs = [model_preprocessing(img_data[1]) for img_data in rotated_imgs]

    # Extract the preprocessed images (rotated_image attribute) and filenames for display
    imgs = [preprocessed.rotated_image for preprocessed in preprocessed_imgs]
    titles = [img_data[0] for img_data in rotated_imgs]

    # Display all preprocessed images with titles
    show_all_imgs(imgs, titles)


# Ansatz 2: Edge Detection (Canny)-> Ausrichtungen der Wände -> Rausfinden zweier Cluster: horizontale & vertikale Wände (k-means) oder horizontale Wände Skalarpordukt=1 und vertikale Wände Skalarprodukt=0
