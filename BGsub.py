# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 00:17:05 2020

@author: chung
"""

import cv2
import numpy as np


# Capture the input frame
def get_frame(cap, scaling_factor=0.5):
    ret, frame = cap.read()
    # Resize the frame
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return frame


if __name__ == '__main__':
    # Initialize the video capture object
    cap = cv2.VideoCapture("data/example_02.mp4")
    # Create the background subtractor object
    bgSubtractor = cv2.createBackgroundSubtractorMOG2()
    history = 100
    # Iterate until the user presses the ESC key
    while True:
        frame = get_frame(cap, 0.5)
        # Apply the background subtraction model to the input frame
        mask = bgSubtractor.apply(frame, learningRate=1.0/history)
        # Convert from grayscale to 3-channel RGB
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        cv2.imshow("mask", mask)
        cv2.imshow('Input frame', frame)
        cv2.imshow('Moving Objects MOG', mask & frame )
        # Check if the user pressed the ESC key
        c = cv2.waitKey(delay=30)
        if c == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

"""
if __name__ == '__main__':
    # Initialize the video capture object
    cap = cv2.VideoCapture("chess.mp4")
    # Create the background subtractor object
    bgSubtractor= cv2.bgsegm.createBackgroundSubtractorGMG()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(3, 3))
    # Iterate until the user presses the ESC key
    while True:
        frame = get_frame(cap, 0.5)
        # Apply the background subtraction model to the input frame
        mask = bgSubtractor.apply(frame)
        # Removing noise from background
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        cv2.imshow('Input frame', frame)
        cv2.imshow('Moving Objects', mask & frame)
        # Check if the user pressed the ESC key
        c = cv2.waitKey(0)
        if c == 27:
            break
    cap.release()
    cv2.destroyAllWindows()"""
