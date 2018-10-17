import cv2
import numpy as np


def get_euler_distance(pt1, pt2):
    return ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[0]) ** 2) ** 0.5

img