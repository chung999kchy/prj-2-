# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 17:28:30 2020

@author: chung
"""
import os
from imutils.video import FPS
import imutils
import cv2
import numpy as np

INPUT_VIDEO = 'data/dynamicBackground/overpass.mp4'
OUTPUT_VIDEO = 'result/overpass_3.mp4'
min_thresh = 0.2
max_thresh = 50
min_area = 1000


def main(video_path):
    out1 = None
    cap = cv2.VideoCapture(video_path)
    S = 0
    BG_gray = None
    index = 0
    motion_frame = 999

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Stopped reading the video (%s)' % video_path)
            break
        index += 1
        print("index", index)
        # resize, convert to Gray and deNoise
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        means = gray.mean()
        if BG_gray is None:
            BG_gray = gray
            continue
        if S == 0:
            (frame_height, frame_width) = frame.shape[:2]
            S = frame_height * frame_width
            print("SSSS" ,S)
            out1 = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                                   (frame_width, frame_height))

        if 0 < index - motion_frame < 5:
            cv2.imshow('a', frame)
            out1.write(frame)
            # cv2.imshow("b", thresh)
            # cv2.imshow("c", BG_gray)
            cv2.imshow("d", frame)
            continue

        # subtraction frame
        frameDelta = cv2.absdiff(BG_gray, gray)
        thresh = cv2.threshold(frameDelta, means // 3, 255, cv2.THRESH_BINARY)[1]
        thresh_percent = (thresh.sum() // 255) / S * 100
        print(thresh.sum() // 255, thresh_percent)
        if min_thresh < thresh_percent < max_thresh:
            motion_frame = index
            # find connectedComponentsWithStats
            thresh = cv2.dilate(thresh, None, iterations=2)
            binary_map = (thresh > 0).astype(np.uint8)
            connectivity = 4  # or whatever you prefer
            ret, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, connectivity, cv2.CV_32S)
            print("labels:", labels)
            print("stats:", stats)
            print("centroids", centroids)
            for i in range(ret):
                if min_area <= stats[i][4] <= S // 2:
                    cv2.rectangle(frame, (stats[i][0], stats[i][1]), (
                        stats[i][0] + stats[i][2], stats[i][1] + stats[i][3]), (0, 255, 0), 2)
            cv2.imshow('a', frame)
            out1.write(frame)
            BG_gray = gray
        elif thresh_percent > max_thresh:  # when camera moving
            BG_gray = gray
        cv2.imshow("b", thresh)
        cv2.imshow("c", BG_gray)
        cv2.imshow("d", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break



    out1.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print('Running frame difference algorithm on %s' % INPUT_VIDEO)
    main(video_path=INPUT_VIDEO)
