import cProfile
from sklearn.cluster import KMeans
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.signal import find_peaks
from semantic_hierarchical_graph import segmentation
from semantic_hierarchical_graph.types.parameter import Parameter
from semantic_hierarchical_graph.types.vector import Vector


class map_preprocessing:
    def __init__(self, img):
        self.gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.grayscales = self.get_grayscales_of_image1()
        self.converted_grayscales_to_black_white_gray = self.convert_grayscales_to_black_white_gray(
            self.gray_image, self.grayscales)
        self.black_pixels_image = self.thresh_black_and_return_as_white(self.converted_grayscales_to_black_white_gray)
        self.white_pixels_image = map_preprocessing.threshold(
            self.converted_grayscales_to_black_white_gray, thresh=250, maxval=255)
        self.dilated_outline = self.dilate_outline(self.black_pixels_image)
        self.removed_incorrect_measured_values = self.remove_incorrect_measured_values()
        self.angles_of_walls, self.lengths_of_walls, self.image_with_orientations = self.get_orientation_of_walls()
        self.rotated_image = self.rotate(int(-(self.find_two_clusters_of_orientations()[0])))

    def add_steps(self, func):
        ...

    def process(self, img):
        ...

    def show_all_imgs(self):
        imgs = []
        img_names = []
        for name, value in vars(self).items():
            if not isinstance(value, list):
                if len(value.shape) == 2:
                    imgs.append(value)
                    img_names.append(name)
        num_rows = 2
        num_cols = len(imgs) // num_rows + 1
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(18, 9))
        for i, (img, name) in enumerate(zip(imgs, img_names)):
            row = i // num_cols
            col = i % num_cols
            axs[row, col].imshow(img, cmap='gray')
            axs[row, col].set_title(name)
            print(str(i) + "/" + str(len(imgs)))
        plt.show()

    def get_grayscales_of_image1(self):
        """returns a sorted list of all used grayscales in the image"""
        # Calculate the histogram
        hist, bins = np.histogram(self.gray_image.ravel(), 255, [0, 255])
        # Count the greayscales used
        counted_grayscales = 0
        for i in hist:
            if i != 0:
                counted_grayscales += 1
        # Sort the histogram bins in descending order
        sorted_bins = np.argsort(-hist)
        # Find the indices of the top three bins
        maxima_indices = sorted_bins[:counted_grayscales]
        # print(len(sorted_bins))
        return maxima_indices

    def get_orientation_of_walls(self, return_img=False):
        # border
        border_img = remove_small_areas_and_write_as_grayscale(self.dilated_outline)
        # Fit a straight line to the smoothed wall line
        lines = cv2.HoughLinesP(border_img, rho=3, theta=np.pi/180, threshold=50, minLineLength=75, maxLineGap=1)
        # Draw the straight line on a black image
        angles = []
        lengths = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                point1 = (x1, self.dilated_outline.shape[0] - y1)
                point2 = (x2, self.dilated_outline.shape[0] - y2)
                if point1[1] < point2[1]:
                    vector = (Vector.from_two_points(point1, point2))
                if point1[1] > point2[1]:
                    vector = (Vector.from_two_points(point2, point1))
                else:  # point1[1] == point2[1]
                    if point1[0] < point2[0]:
                        vector = (Vector.from_two_points(point2, point1))
                    else:
                        vector = (Vector.from_two_points(point2, point1))
                cv2.line(border_img, (x1, y1), (x2, y2), (100), 2)
                # Calculate the orientation in radians
                angle_radians = math.atan2(vector.y, vector.x)  # firstly y then x
                # Convert the orientation to degrees
                angle_degree = round(math.degrees(angle_radians), 0)
                # write degrees between 145° and 180° in the value range -45° to 0°
                # if angle_degree > 145:
                #   angle_degree = 180-angle_degree
                angles.append(angle_degree)
                # calculate the length (for later usage)
                length = int(math.sqrt((vector.x)**2 + (vector.y)**2))
                lengths.append(length)
                # Testing the angles: print the degree in the wall and check on plausibility
                pos = ((x1 + x2)//2, (y1 + y2)//2)
                cv2.putText(border_img, str(angle_degree), pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else:
            img_copy = self.dilated_outline.copy()
            no_result = cv2.putText(img_copy, "No walls detected", (20, 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            return angles, lengths, no_result
        return angles, lengths, border_img

    @staticmethod
    def find_bounding_box(threshed_image):
        # Find the coordinates of all nonzero pixels in the image
        nonzero_pixels = np.nonzero(threshed_image)
        # Get the minimum and maximum x and y coordinates of the nonzero pixels
        min_x = np.min(nonzero_pixels[1])
        max_x = np.max(nonzero_pixels[1])
        min_y = np.min(nonzero_pixels[0])
        max_y = np.max(nonzero_pixels[0])

        # Return the bounding box coordinates as a tuple
        return (min_x, min_y, max_x - min_x, max_y - min_y)

    def find_two_clusters_of_orientations(self):

        angles_weighted = []
        for i in range(len(self.angles_of_walls)):
            for l in range(0, self.lengths_of_walls[i], 10):
                angles_weighted.append(self.angles_of_walls[i])
        print(self.lengths_of_walls)
        print(angles_weighted)
        # use KMeans clustering to find two dominant clusters
        kmeans = KMeans(n_clusters=2).fit(np.array(angles_weighted).reshape(-1, 1))

        # get the cluster labels and centroids
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        print("Cluster labels: ", labels)
        print("Cluster centroids: ", centroids)
        return centroids

    def rotate(self, degree):
        # Get the dimensions of the image
        height, width = self.gray_image.shape[:2]
        # expand the image with a padding of the size of the image's diagonal length. This length is choosed randomly. It just have to be large enough
        diagonal = int((height**2 + width**2)**0.5)
        padded_img = cv2.copyMakeBorder(self.removed_incorrect_measured_values, diagonal, diagonal, diagonal,
                                        diagonal, cv2.BORDER_CONSTANT, value=[205, 205, 205])
        # calculate the RotationMatrix of the expanded/ padded image
        new_height, new_width = padded_img.shape[:2]
        center = (new_width / 2, new_height / 2)
        M = cv2.getRotationMatrix2D(center, degree, 1.0)
        # rotate the padded image
        converted_img = cv2.convertScaleAbs(padded_img)
        rotated_img = cv2.warpAffine(converted_img, M, (new_width, new_height),
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=205)
        # as it seems, the rotaed img is blurred. So this will convert the image to three pixel values
        rotated_img = self.convert_grayscales_to_black_white_gray(rotated_img, self.grayscales)
        # Now the padded image is larger than it should be. In the next steps we crop the image to a nice size
        # Threshold to create a binary image
        thresh = cv2.threshold(rotated_img, 250, 255, cv2.THRESH_BINARY)[1]
        # Get bounding box
        x, y, w, h = find_bounding_box(thresh)
        # Draw rectangle on image: just for demonstration
        # cv2.rectangle(rotated_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Crop bbox for perfect image size
        crop_x1, crop_y1 = max(0, x-20), max(0, y-20)
        crop_x2, crop_y2 = min(rotated_img.shape[1], x+w+20), min(rotated_img.shape[0], y+h+20)

        return rotated_img[crop_y1:crop_y2, crop_x1:crop_x2]

    @ staticmethod
    def threshold(img, thresh=100, maxval=255):
        ret, thresh_img = cv2.threshold(img, thresh, maxval, cv2.THRESH_BINARY)
        return thresh_img

    @ staticmethod
    def thresh_black_and_return_as_white(img):
        # thresh all black pixels (recognised barriers)
        ret, thresh_black_pixels = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
        black_dots_inversed = cv2.subtract(255, thresh_black_pixels)
        return black_dots_inversed

    @ staticmethod
    def dilate_outline(img, kernel_size=8):
        """Black pixels are recorded barriers at this location and represents the wall or other obstacles.
        Some pixels aren't recorded properly and little gaps are in the wall. By dilation the gaps will be filled/ continous lines
        Returns an image with dilated black pixels"""
        # expand/ dialte the black pixels to get continous lines
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_blackdots = cv2.dilate(img, kernel, iterations=1)
        return dilated_blackdots

    @ staticmethod
    def convert_grayscales_to_black_white_gray(img, grayscales):
        """Online maps/ ressources often don't use just three pixel values:
        This function will write the brigtest values as white, the darkest values as black and the values between as gray"""
        # Get the highest and lowest color values
        min_value = min(grayscales)
        max_value = max(grayscales)
        # Overwrite the highest value with 254, the lowest with 0, and the rest with 205
        # Iterate through the data set and overwrite values
        for row in range(len(img)):
            for pixel in range(len(img[row])):
                if img[row][pixel] > max_value - 10:
                    img[row][pixel] = 254
                elif img[row][pixel] < min_value + 10:
                    img[row][pixel] = 0
                else:
                    img[row][pixel] = 205
        return img

    @ staticmethod
    def remove_small_areas_and_write_as_grayscale(img, grayscale_value=255, min_area=400):
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

    def remove_incorrect_measured_values(self):
        """On the origin recorded map are long thin lines. These lines are created when, for example, the laser passes through a window and doesn't hit an object"""
        islands = cv2.subtract(self.white_pixels_image, self.dilated_outline)
        # create image of the main map without thin lines
        img_main_map_without_boundaries = remove_small_areas_and_write_as_grayscale(islands)
        # create image of the unrecorded area
        ret, img_of_unknown_area = cv2.threshold(img_main_map_without_boundaries, 0, 255, cv2.THRESH_BINARY_INV)
        ret, img_of_unknown_area_scaled_to_230 = cv2.threshold(img_of_unknown_area, 0, 230, cv2.THRESH_BINARY)
        # create image of the border
        img_of_outline = remove_small_areas_and_write_as_grayscale(self.dilated_outline)
        # merge the three images
        img_main_map_and_unknown_area = cv2.add(img_main_map_without_boundaries, img_of_unknown_area_scaled_to_230)
        result = cv2.subtract(img_main_map_and_unknown_area, img_of_outline)
        return result


def map_preprocessing_pipeline(img):
    """1) remove noise 2) remove thin lines 3) rotate: dilate extracted outlines -> fit lines/extract walls -> get orientation of line -> clustering"""
    # some white pixels are outside the building. This is corrected in the following function:
    image_without_white_pixel_outside = remove_incorrect_measured_values(img)
    segmentation.show_imgs(image_without_white_pixel_outside)
    # get lines of straight and long borders
    # write_walls_as_lines(img)
    get_orientation_of_walls(img)


def show_all_imgs(imgs):
    plt.figure(figsize=(30, 20))
    for i, img in enumerate(imgs):
        plt.subplot(4, int(len(imgs)/4)+1, i+1)
        plt.imshow(img, cmap='gray')
    plt.show()


def optimise_the_parameters():
    ...


def get_grayscales_of_image(img):
    """returns a sorted list of all used grayscales in the image"""
    # convert to grayscale if not already done
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    # Calculate the histogram
    hist, bins = np.histogram(gray.ravel(), 255, [0, 255])
    # Count the greayscales used
    counted_grayscales = 0
    for i in hist:
        if i != 0:
            counted_grayscales += 1
    # Sort the histogram bins in descending order
    sorted_bins = np.argsort(-hist)
    # Find the indices of the top three bins
    maxima_indices = sorted_bins[:counted_grayscales]
    # print(len(sorted_bins))
    return maxima_indices


def convert_grayscales_to_black_white_gray(img):
    """Online maps/ ressources often don't use just three pixel values:
    This function will write the brigtest values as white, the darkest values as black and the values between as gray"""
    # convert to grayscale
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    # Find the indices of the top three bins
    maxima_indices = get_grayscales_of_image(img)[:3]
    print(maxima_indices)
    # Get the highest and lowest color values
    min_value = min(maxima_indices)
    max_value = max(maxima_indices)
    # Overwrite the highest value with 254, the lowest with 0, and the rest with 205
    # Iterate through the data set and overwrite values
    for row in range(len(gray)):
        for pixel in range(len(gray[row])):

            if gray[row][pixel] > max_value - 10:
                gray[row][pixel] = 254
            elif gray[row][pixel] < min_value + 10:
                gray[row][pixel] = 0
            else:
                gray[row][pixel] = 205
    return gray


def dilate_outline(img, kernel_size=8):
    """Black pixels are recorded barriers at this location and represents the wall or other obstacles.
    Some pixels aren't recorded properly and little gaps are in the wall. By dilation the gaps will be filled/ continous lines
    Returns an image with dilated black pixels"""
    # convert to grayscale
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    # convert to three grayscales, so that thresholding recognise the black
    if len(get_grayscales_of_image(img)) != 3:
        gray = convert_grayscales_to_black_white_gray(img)
    # thresh all black pixels (recognised barriers)
    ret, thresh_black_pixels = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    black_dots_inversed = cv2.subtract(255, thresh_black_pixels)
    # expand/ dialte the black pixels to get continous lines
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_blackdots = cv2.dilate(black_dots_inversed, kernel, iterations=1)
    return dilated_blackdots


def remove_small_areas_and_write_as_grayscale(img, grayscale_value=255, min_area=400):
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


def seperate_thin_lines_from_building(img):
    """Dilate the border and subtract it from the main map. Consequencely the thin lines, when the laser passes through a window, will be seperated from the main map"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    continous_outline_img = dilate_outline(gray, kernel_size=8)
    # thresh all  white pixels
    ret, thresh_white_pixels = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    # subtract the black and grey pixels from the white pixels, so the thin lines are seperated from the true map
    islands = cv2.subtract(thresh_white_pixels, continous_outline_img)
    return islands


def remove_incorrect_measured_values(img):
    """On the origin recorded map are long thin lines. These lines are created when, for example, the laser passes through a window and doesn't hit an object"""
    # create image of the main map without thin lines
    img_main_map_without_boundaries = remove_small_areas_and_write_as_grayscale(seperate_thin_lines_from_building(img))
    # create image of the unrecorded area
    ret, img_of_unknown_area = cv2.threshold(img_main_map_without_boundaries, 0, 255, cv2.THRESH_BINARY_INV)
    ret, img_of_unknown_area_scaled_to_230 = cv2.threshold(img_of_unknown_area, 0, 230, cv2.THRESH_BINARY)
    # create image of the border
    img_of_outline = remove_small_areas_and_write_as_grayscale(dilate_outline(img))
    # merge the three images
    img_main_map_and_unknown_area = cv2.add(img_main_map_without_boundaries, img_of_unknown_area_scaled_to_230)
    result = cv2.subtract(img_main_map_and_unknown_area, img_of_outline)
    return result


def get_orientation_of_walls(img, return_img=False):
    # border
    border_img = remove_small_areas_and_write_as_grayscale(dilate_outline(img, kernel_size=4), min_area=500)
    # Fit a straight line to the smoothed wall line
    lines = cv2.HoughLinesP(border_img, rho=3, theta=np.pi/180, threshold=50, minLineLength=75, maxLineGap=1)
    # Draw the straight line on a black image
    angles = []
    lengths = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            point1 = (x1, img.shape[0] - y1)
            point2 = (x2, img.shape[0] - y2)
            if point1[1] < point2[1]:
                vector = (Vector.from_two_points(point1, point2))
            if point1[1] > point2[1]:
                vector = (Vector.from_two_points(point2, point1))
            if point1[1] == point2[1]:
                if point1[0] < point2[0]:
                    vector = (Vector.from_two_points(point2, point1))
                else:
                    vector = (Vector.from_two_points(point2, point1))
            cv2.line(border_img, (x1, y1), (x2, y2), (100), 2)
            # Calculate the orientation in radians
            angle_radians = math.atan2(vector.y, vector.x)  # firstly y then x
            # Convert the orientation to degrees
            angle_degree = round(math.degrees(angle_radians), 0)
            # write degrees between 145° and 180° in the value range -45° to 0°
            # if angle_degree > 145:
            #   angle_degree = 180-angle_degree
            angles.append(angle_degree)
            # calculate the length (for later usage)
            length = int(math.sqrt((vector.x)**2 + (vector.y)**2))
            lengths.append(length)
            # Testing the angles: print the degree in the wall and check on plausibility
            pos = ((x1 + x2)//2, (y1 + y2)//2)
            cv2.putText(border_img, str(angle_degree), pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        if return_img == True:
            return border_img
    else:
        img_copy = img.copy()
        no_result = cv2.putText(img_copy, "No walls detected", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        return no_result
    return angles, lengths


def find_two_clusters_of_orientations(angles, lengths):
    angles_weighted = []
    for i in range(len(angles)):
        for l in range(0, lengths[i], 10):
            angles_weighted.append(angles[i])
    print(lengths)
    print(angles_weighted)
    # use KMeans clustering to find two dominant clusters
    kmeans = KMeans(n_clusters=2).fit(np.array(angles_weighted).reshape(-1, 1))

    # get the cluster labels and centroids
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    print("Cluster labels: ", labels)
    print("Cluster centroids: ", centroids)
    return centroids


def rotate(img, degree):
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
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=205)
    # as it seems, the rotaed img is blurred. So this will convert the image to three pixel values
    rotated_img = convert_grayscales_to_black_white_gray(rotated_img)
    # Now the padded image is larger than it should be. In the next steps we crop the image to a nice size
    # Threshold to create a binary image
    thresh = cv2.threshold(rotated_img, 250, 255, cv2.THRESH_BINARY)[1]
    # Get bounding box
    x, y, w, h = find_bounding_box(thresh)
    # Draw rectangle on image: just for demonstration
    # cv2.rectangle(rotated_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # Crop bbox for perfect image size
    crop_x1, crop_y1 = max(0, x-20), max(0, y-20)
    crop_x2, crop_y2 = min(rotated_img.shape[1], x+w+20), min(rotated_img.shape[0], y+h+20)

    return rotated_img[crop_y1:crop_y2, crop_x1:crop_x2]


def make_image_to_three_values(img):
    data = img.reshape((-1, 3))
    # Perform k-means clustering with k=3
    kmeans = KMeans(n_clusters=3, random_state=0).fit(data.reshape(-1, 1))
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    # Assign values to each cluster
    values = np.array([0, 205, 254])
    # Map each pixel to its corresponding value
    mapped_values = np.zeros_like(labels)
    mapped_values[labels == np.argmax(centers[:, 0])] = values[2]
    mapped_values[labels == np.argmin(centers[:, 0])] = values[0]
    mapped_values[(labels != np.argmax(centers[:, 0])) & (labels != np.argmin(centers[:, 0]))] = values[1]
    # Map each pixel to its corresponding value
    mapped_pixels = mapped_values.reshape(img.shape)

    return mapped_pixels


def find_bounding_box(threshed_image):
    # Find the coordinates of all nonzero pixels in the image
    nonzero_pixels = np.nonzero(threshed_image)
    # Get the minimum and maximum x and y coordinates of the nonzero pixels
    min_x = np.min(nonzero_pixels[1])
    max_x = np.max(nonzero_pixels[1])
    min_y = np.min(nonzero_pixels[0])
    max_y = np.max(nonzero_pixels[0])

    # Return the bounding box coordinates as a tuple
    return (min_x, min_y, max_x - min_x, max_y - min_y)


def show_andy_the_new_functions():
    img1 = cv2.imread('data\\benchmark_maps\\iras_slam2.png')
    img2 = cv2.imread('data\\benchmark_maps\\aces_austin.png')
    print(get_grayscales_of_image(img1))
    # show from many grayscales to three values: convert_grayscales_to_black_white_gray
    segmentation.show_imgs(img2, convert_grayscales_to_black_white_gray(img2))
    # show dilate outline
    segmentation.show_imgs(img1, dilate_outline(img1))
    segmentation.show_imgs(img2, dilate_outline(img2))
    # show removed thin lines
    segmentation.show_imgs(img1, remove_incorrect_measured_values(img1))
    # show rotation png in Ordner

    # show approx poly approach results
    # show walls approach results
    # demonstrate the 180 entspricht 0 problem


def create_rotated_img():
    import random
    imgs = []
    # imgs.append(cv2.imread('data\\benchmark_maps\\acapulco_convention_center.png'))
    # imgs.append(cv2.imread('data\\benchmark_maps\\intel_research_lab.png'))
    # imgs.append(cv2.imread('data\\benchmark_maps\\iras_slam1.png'))
    imgs.append(cv2.imread('data\\benchmark_maps\\ryu.png'))
    imgs.append(cv2.imread('data\\benchmark_maps\\orebro_small.png'))
    imgs.append(cv2.imread('data\\benchmark_maps\\scsail.png'))
    imgs.append(cv2.imread('data\\benchmark_maps\\seattle_r.png'))
    # imgs.append(cv2.imread('data\\benchmark_maps\\sedmonton.png'))
    # imgs.append(cv2.imread('data\\benchmark_maps\\sfr_campus_20040714.png'))
    # imgs.append(cv2.imread('data\\benchmark_maps\\sfr101.carmen.png'))
    # imgs.append(cv2.imread('data\\benchmark_maps\\sMIT_Infinite_Corridor_2002_09_11_same_floor.png'))
    rotated_imgs = []
    for i, img in enumerate(imgs):
        random_number = random.uniform(0, 360)
        segmentation.show_imgs(make_image_to_three_values(img))
        con = make_image_to_three_values(img)
        rotated = rotate(con, random_number)
        rotated_imgs.append(rotated)
    return rotated_imgs


def predict_list_of_images():
    imgs = create_rotated_img()
    predicted_images = []
    for img in imgs:
        predicted_images.append(img)
        predicted_images.append(get_orientation_of_walls(img, return_img=True))
    show_all_imgs(predicted_images)


def testing_speed():
    """shows me the runtime of the functions"""
    cProfile.run('map_preprocessing(img1)')


if __name__ == '__main__':
    img1 = cv2.imread('data\\benchmark_maps\\iras_slam2.png')
    img2 = cv2.imread('data\\benchmark_maps\\aces_austin.png')
    img3 = cv2.imread('data\\benchmark_maps\\ryu.png')
    params = Parameter("config/ryu_params.yaml").params
    preprocessed = map_preprocessing(img1)
    preprocessed.show_all_imgs()

    # gray = preprocessed.black_pixels_image
    # dilated = preprocessed.removed_incorrect_measured_values
    # print(preprocessed.grayscales)
    # segmentation.show_imgs(img1, dilated)
    # predict_list_of_images()
    # segmentation.show_imgs(img3, make_image_to_three_values(img3))
    # segmentation.show_imgs(rotate(convert_grayscales_to_black_white_gray(img3), 40))
# Ansatz 2: Edge Detection (Canny)-> Ausrichtungen der Wände -> Rausfinden zweier Cluster: horizontale & vertikale Wände (k-means) oder horizontale Wände Skalarpordukt=1 und vertikale Wände Skalarprodukt=0

"""anderes Clustering verwenden

Mittelwert zischen den Peaks des Histogramms nehmen

Parameter in yaml (eigene Aufnahmen und online Ressourcen)

Drehung nach Mehrheit orientieren

Grundriss durch pointcloud libary"""


def padding(img, padding_size):
    # make a padding and substract it from origin --> thin lines will be deleted
    # Define the color of interest
    color_lower = np.array(0.5)
    color_upper = np.array(0.5)
    # Create a mask that identifies the pixels of the color of interest
    mask = cv2.inRange(img, color_lower, color_upper)
    # Apply a dilation operation to expand the mask
    kernel = np.ones((padding_size, padding_size), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=2)
    # Apply the dilated mask to the original image to add padding around the color of interest
    padding = cv2.bitwise_and(img, img, mask=dilated_mask)
    img = cv2.subtract(img, padding)
    return img


def detect_minimal_bboxes(img):
    "locate and draw in multiple bounding boxes. Use Case: draw a minimal bbox around the rooms of the map and calculate the orientation"
    imgs = []
    img_with_bboxes = img.copy()
    list_of_orientations_in_degree = []
    list_of_orientations_in_degree_normalized = []

    for instance in range(img.max()):
        # wathershed algorithm writes the unsure area to -1, to stay in the value range "+1"
        imgs.append(np.uint8(img.copy())+1)
        print(instance)
        for row in range(len(imgs[instance])):
            for pixel in range(len(imgs[instance][row])):
                if imgs[instance][row][pixel] != instance:
                    imgs[instance][row][pixel] = 0
        # draw the minimal bbox in every picture
        contours, _ = cv2.findContours(imgs[instance], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Loop over all contours and find the minimum bounding box
        for contour in contours:
            # Calculate the minimum bounding box for the contour
            rect = cv2.minAreaRect(contour)
            # Draw the minimum bounding box on the image
            box = cv2.boxPoints(rect)
            box = box.astype(int)
            cv2.drawContours(imgs[instance], [box], 0, (255, 0, 255), 5)
            # Draw all bboxes in one picture
            cv2.drawContours(img_with_bboxes, [box], 0, (255, 0, 255), 5)
            # Save the orientation of the bbox in a list and represent it in the picture
            list_of_orientations_in_degree.append(rect[2])  # rect[2] contains the degreee/rad of the bbox
            list_of_orientations_in_degree_normalized.append(abs(rect[2]-45))
            cv2.putText(imgs[instance], str(round(abs(rect[2]-45)-45, 1)), (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), thickness=2)
            print(list_of_orientations_in_degree)
    # calculate the mean degree of the orientation of the rooms
    # to do/try: 1) correct the outlier (look 3) and 2) weight the orientation with the area and 3) if area bbox has a big differece to real area room, then do not consider the orientation of the bbox
    imgs.append(img)
    imgs.append(img_with_bboxes)
    show_all_imgs(imgs)
    return img


def map_preparation(img):
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # convert to binary
    ret, thresh = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    deleted_padding_img = padding(thresh, padding_size=3)
    # Apply median filter to remove noise
    denoised = cv2.medianBlur(deleted_padding_img, 5)  # increase the kernel size for stronger denoising

    # delete leftover dots
    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(denoised, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Define the minimum area of a contour to keep
    min_area = 500
    # Filter contours by area and remove small contours
    filtered_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        if area <= min_area:
            filtered_contours.append(cnt)
    # Draw the filtered contours on a new blank image
    contours_img = np.zeros_like(denoised)
    cv2.drawContours(contours_img, filtered_contours, -1, 1, 2)
    cv2.fillPoly(contours_img, filtered_contours, 1)
    deleted_leftofer_dots_img = cv2.subtract(denoised, contours_img)

    # Add back the unwanted deleted padding
    inverse_deleted_leftofer_dots_img = 1 - deleted_leftofer_dots_img
    inverse_restored_padding_img = padding(inverse_deleted_leftofer_dots_img, padding_size=3)
    restored_padding_img = 1 - inverse_restored_padding_img

    # Apply median filter to remove noise
    # increase the kernel size for stronger denoising
    denoised_restored_padding_img = cv2.medianBlur(restored_padding_img, 5)

    return denoised_restored_padding_img


def detect_walls(img):
    # canny is dosn't represent the real border. The black pixels does
    # Apply edge detection to extract wall line
    # edges = cv2.Canny(img, 50, 150)
    # Smooth the wall line
    smoothed = cv2.GaussianBlur(edges, (5, 5), 0)
    # Fit a straight line to the smoothed wall line
    lines = cv2.HoughLinesP(smoothed, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
    # Draw the straight line on the original image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return img


def extract_walls(img):
    img_corrected = remove_incorrect_measured_values(img)
    # make binary
    ret, img_corrected_binary = cv2.threshold(img_corrected, 250, 255, cv2.THRESH_BINARY)
    # dialte to fill in the gaps of missing data and to overwrite the dynamic objects
    kernel_size = 9
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_room = cv2.dilate(img_corrected_binary, kernel, iterations=1)

    edges = cv2.Canny(img_corrected_binary, 200, 230)

    # Smooth the wall line
    smoothed_edges = cv2.GaussianBlur(edges, (5, 5), 0)

    # expand/ dialte the black pixels to get continous lines
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_smoothed, _ = cv2.findContours(smoothed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw original contours on image
    img_copy = img.copy()
    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    cv2.drawContours(img_copy, contours_smoothed, -1, (0, 255, 0), 2)

    # Approximate contours and draw them on image
    for contour in contours_smoothed:
        epsilon = 0.003 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(img, [approx], -1, (0, 0, 255), 2)

    # calculate the orientations of the walls
    angles = []
    lengths = []
    for point in range(len(approx)-1):
        # get the vector from two points
        vector = (Vector.from_two_points(approx[point][0], approx[point + 1][0]))
        # Calculate the orientation in radians
        angle_radians = math.atan2(vector.y, vector.x)  # als erstes y dann x
        # Convert the orientation to degrees
        angle_degree = 180 - ((round(math.degrees(angle_radians), 0) + 180) % 180)
        angles.append(angle_degree)
        # calculate the length (for later usage)
        length = int(math.sqrt((vector.x)**2 + (vector.y)**2))
        lengths.append(length)
        # Testing the angles: print the degree in the wall and check on plausibility
        pos = (approx[point][0] + approx[point + 1][0])//2
        cv2.putText(img, str(angle_degree), pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    # to weigth the angles with the length of the wall add to the angles list copies of the angles in relation to its length -> for better clustering
    # Bsp: angles=[40, 20, 50], lengths = [60, 90, 20] -> angles_weighted[40, 40, 40, 40, 40, 40, 20, 20, 20, 20, 20, 20, 20, 20, 20, 50, 50]
    angles_weighted = []
    for i in range(len(angles)):
        for l in range(0, lengths[i], 10):
            angles_weighted.append(angles[i])
    print(lengths)
    print(angles_weighted)
    # use KMeans clustering to find two dominant clusters
    kmeans = KMeans(n_clusters=2).fit(np.array(angles_weighted).reshape(-1, 1))

    # get the cluster labels and centroids
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    print("Cluster labels: ", labels)
    print("Cluster centroids: ", centroids)
    # rotated Image:
    # Get the dimensions of the image
    height, width = img.shape[:2]
    print(centroids[1])
    # Calculate the rotation matrix
    # Better: if degreee between both centroids isn't 90°-> find the degree so that both centroids are roteted with an equal error. Otherwise the error
    rotation_matrix_1 = cv2.getRotationMatrix2D((width/2, height/2), -(int(centroids[0])), 1)
    rotation_matrix_2 = cv2.getRotationMatrix2D((width/2, height/2), -(int(centroids[1])), 1)
    # Specify the padding parameters
    border_size = 100
    border_value = [205, 205, 205]
    # Apply the rotation to the image
    # rotated = cv2.warpAffine(img, rotation_matrix, (width, height))

    rotated_1 = cv2.warpAffine(img, rotation_matrix_1, (width + 2*border_size, height + 2*border_size),
                               borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)
    rotated_2 = cv2.warpAffine(img, rotation_matrix_2, (width + 2*border_size, height + 2*border_size),
                               borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)
    segmentation.show_imgs(img, rotated_2)
    return img


def testing_the_value_for_threshold(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh_hold_tests = []
    a = 0.0
    for i in range(0, 200, 20):
        # convert to binary
        a = i/10
        ret, thresh = cv2.threshold(gray, i, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.putText(thresh, str(a), (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), thickness=2)
        thresh_hold_tests.append(thresh)
    show_all_imgs(thresh_hold_tests)


def testing_the_noise_removal(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel_tests = []
    for kernel in range(1, 20, 2):
        # Apply median filter to remove noise
        denoised = cv2.medianBlur(gray, kernel)  # increase the kernel size for stronger denoising
        cv2.putText(denoised, "Kernel size: " + str(kernel), (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), thickness=2)
        kernel_tests.append(denoised)
    show_all_imgs(kernel_tests)


def testing_padding_arround_color(img):
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # convert to binary
    ret, thresh = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    segmentation.show_imgs(thresh)
    # Define the color of interest in HSV color space
    color_lower = np.array(0.5)
    color_upper = np.array(0.5)
    # Create a mask that identifies the pixels of the color of interest
    mask = cv2.inRange(thresh, color_lower, color_upper)
    # Apply a dilation operation to expand the mask
    padding_size = 5
    kernel = np.ones((padding_size, padding_size), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    # Apply the dilated mask to the original image to add padding around the color of interest
    padding = cv2.bitwise_and(thresh, thresh, mask=dilated_mask)
    result = thresh - padding
    segmentation.show_imgs(result)


def remove_inadequately_recorded_rooms(img):
    dilated_outlines = dilate_outline(img)
    connected_contours = connect_contours(dilated_outlines)
    removed_rooms = remove_small_areas(connected_contours)
    return removed_rooms


def connect_contours(img):
    """maybe not necessary. recorded black lines should be enough to calculate the orientation.
    But not the canny of the recorded white area should be use. then we make walls, where no walls are"""
    img_copy = img.copy()

    # Find contours
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Approximate contours and draw them on image
    for contour in contours:
        epsilon = 0.003 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(img, [approx], -1, (0, 0, 255), 2)
    segmentation.show_imgs(img)
    """# Find the endpoints of the contour
    endpoints = []
    # Loop over each contour
    for cnt in contours:
        print(cnt)
        for i in range(len(cnt)):
            if i == 0 or i == len(cnt)-1:
                endpoints.append(cnt[i])
            elif cnt[i-1][0][0] != cnt[i][0][0] and cnt[i+1][0][0] != cnt[i][0][0]:
                endpoints.append(cnt[i])
    test2 = cv2.drawContours(img, endpoints, -1, (0, 255, 0), 3)
    # Loop over each endpoint
    for ep1 in endpoints:
        # Calculate the Euclidean distance to all other points on the contour
        distances = []
        for ep2 in endpoints:
            # print("1")
            if np.array_equal(ep1, ep2) == False:
                dist = np.linalg.norm(np.array(ep1[0])-np.array(ep2[0]))
                distances.append(dist)
                # print("2")

        # Find the point on the contour with the minimum Euclidean distance
        min_dist_index = np.argmin(distances)
        # print(min_dist_index)
        min_dist_pt = endpoints[min_dist_index]
        # print(min_dist_pt)
        # Connect the endpoint to the identified point

        cv2.line(img, tuple(ep1[0]), tuple(min_dist_pt[0]), (255), 2)
        test = cv2.circle(img, tuple(ep1[0]), radius=0, color=(255), thickness=-1)"""
    print(tuple(ep1[0]))
    print("ep1")
    print(ep1)
    segmentation.show_imgs(img, test)
    return img
