import cv2
import numpy as np
from matplotlib import pyplot as plt
import random

x = {
    0: [-1, -1], 1: [-1, 0], 2: [-1, 1],
    3: [0, -1], 4: [0, 1],
    5: [1, -1], 6: [1, 0], 7: [1, 1]
}


def randomPixel(image):
    img = None
    img = image.copy()
    height, width = image.shape[:2]
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            y = random.randint(0, 7)
            img[i][j] = image[i + x[y][0]][j + x[y][1]]
    return img
