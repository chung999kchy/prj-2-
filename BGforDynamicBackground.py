import os
import imutils
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from imutils.video import FPS
import nhap as r
import random

N1 = 8
m = 5
tau_min = 5
N = 40

INPUT_VIDEO = 'data/dynamicBackground/overpass.mp4'
OUTPUT_VIDEO = 'result/overpass_2.mp4'


def main(video_path):
    index = 0
    out1 = None
    cap = cv2.VideoCapture(video_path)
    S = 0
    fps = FPS().start()
    BG = []

    while True:
        ret, frame = cap.read()
        index += 1
        if not ret:
            print('Stopped reading the video (%s)' % video_path)
            break
        # resize, convert to Gray and deNoise
        frame = imutils.resize(frame, width=500)
        if S == 0:
            (frame_height, frame_width) = frame.shape[:2]
            S = frame_height * frame_width
            out1 = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                                   (frame_width, frame_height))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if index <= N1:
            for i in range(m):
                img = r.randomPixel(gray)
                BG.append(img)
            continue
        # lbp = local_binary_pattern(gray, 8, 1, method="uniform")
        x1 = random.randint(0, N-1)
        x2 = random.randint(0, N-1)
        frameDelta1 = cv2.absdiff(BG[x1], gray)
        frameDelta2 = cv2.absdiff(BG[x2], gray)
        thresh1 = cv2.threshold(frameDelta1, 40, 255, cv2.THRESH_BINARY)[1]
        thresh2 = cv2.threshold(frameDelta2, 40, 255, cv2.THRESH_BINARY)[1]
        thresh = thresh1 & thresh2


        cv2.imshow("a", frame)
        cv2.imshow("b", thresh)

        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print('Running frame difference algorithm on %s' % INPUT_VIDEO)
    main(video_path=INPUT_VIDEO)
