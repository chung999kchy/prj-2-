import numpy as np
import cv2
import math
import statistics

# Create K is value of displacement Li
K = 1
K1 = 0.001


def displacement(a, b, c, d):
    dic = (a - c) ** 2 + (b - d) ** 2
    return math.sqrt(dic)


# params for ShiTomasi corner dectection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=2,
                      blockSize=7)

# Params for Lucas Kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=4,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = (255, 255, 0)


def calOp(last_frame, frame, thresh):
    # Define the codec and create VideoWriter object.The output is stored in 'output3.avi' file.

    p0 = cv2.goodFeaturesToTrack(thresh, mask=None, **feature_params)
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(last_frame, frame, p0, None, **lk_params)
    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    check = True
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        frame = cv2.line(frame, (a, b), (c, d), color, 2)

    # find connectedComponentsWithStats
    thresh = cv2.dilate(thresh, None, iterations=2)
    binary_map = (thresh > 0).astype(np.uint8)
    connectivity = 4  # or whatever you prefer
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, connectivity, cv2.CV_32S)
    for i in range(ret):
        if stats[i][4] > 500:
            cv2.rectangle(frame, (stats[i][0], stats[i][1]), (
                stats[i][0] + stats[i][2], stats[i][1] + stats[i][3]), (0, 255, 0), 2)

    cv2.imshow("show", frame)
    return check
