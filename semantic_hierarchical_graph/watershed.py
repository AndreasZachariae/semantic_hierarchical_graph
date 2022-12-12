import numpy as np
import cv2
from matplotlib import pyplot as plt
import largestinteriorrectangle as lir


def marker_controlled_watershed(img: np.ndarray):
    # convert to binary
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    # 0.5 * dist_transform.max()
    # TODO: find a better threshold value
    ret, sure_fg = cv2.threshold(dist_transform, 30, 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]

    # show_imgs(img, markers)

    return markers


def largest_rectangle_per_region(markers: np.ndarray):
    # print(markers.max())
    # region_bool = np.where(markers == 10, True, False)

    for i in range(2, markers.max() + 1):
        region_bool = np.where(markers == i, True, False)
        region = region_bool.astype("uint8") * 255
        contours, _ = cv2.findContours(region, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contour = contours[0][:, 0, :]
        rectangle = lir.lir(region_bool, contour)

        markers = cv2.rectangle(markers, rectangle, (22), 2)

    show_imgs(markers)


def show_imgs(img: np.ndarray, img_2: np.ndarray = None):
    if img_2 is not None:
        plt.subplot(211), plt.imshow(img)
        plt.subplot(212), plt.imshow(img_2)
    else:
        plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    img = cv2.imread('data/map_benchmark_ryu.png')

    markers = marker_controlled_watershed(img)
    largest_rectangle_per_region(markers)
