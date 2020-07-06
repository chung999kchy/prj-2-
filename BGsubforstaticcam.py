# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 17:28:30 2020

@author: chung
"""
import imutils
import cv2
import numpy as np
from imutils.video import FPS
from skimage.feature import local_binary_pattern
import calOptical as cal

INPUT_VIDEO = 'data/dynamicBackground/overpass.mp4'
OUTPUT_VIDEO = 'result/overpass_2.mp4'
OUTPUT_VIDEO2 = 'result/overpass_den.mp4'
min_thresh_percent = 1
max_thresh_percent = 50


def main(video_path):
    index = 0
    out1 = None
    out2 = None
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_FPS, 30)
    bgSubtractor = cv2.createBackgroundSubtractorMOG2()
    bgSubtractor.setVarThreshold(40)
    S = 0
    thresh = None
    fgmask = None
    fps = FPS().start()
    last_frame = None
    while True:
        ret, frame = cap.read()
        index += 1
        if not ret:
            print('Stopped reading the video (%s)' % video_path)
            break
        # resize, convert to Gray and deNoise
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if S == 0:
            (frame_height, frame_width) = frame.shape[:2]
            S = frame_height * frame_width * 255
            out1 = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                                   (frame_width, frame_height))

            out2 = cv2.VideoWriter(OUTPUT_VIDEO2, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                                   (frame_width, frame_height))
        if last_frame is None:
            last_frame = gray
        # subtraction frame
        fgmask = bgSubtractor.apply(gray, fgmask, learningRate=-1)
        fgmask = cv2.GaussianBlur(fgmask, (21, 21), 0)
        thresh = cv2.threshold(fgmask, 60, 255, cv2.THRESH_BINARY)[1]
        counter = np.sum(thresh)
        thresh_percent = counter * 100 / S
        print(thresh_percent)
        if min_thresh_percent < thresh_percent < max_thresh_percent:
            # if cal.calOp(last_frame, gray, thresh):
            cv2.imshow('Video output', frame)
            out1.write(frame)
        out2.write(thresh)
        cv2.imshow("Segmap", thresh)
        cv2.imshow("Video goc", frame)
        key = cv2.waitKey(60)
        last_frame = gray
        if key == 27:
            break
        fps.update()

    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    out1.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print('Running frame difference algorithm on %s' % INPUT_VIDEO)
    main(video_path=INPUT_VIDEO)
