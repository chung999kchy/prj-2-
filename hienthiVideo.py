# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 19:33:02 2019

@author: chung
"""

import cv2

cap = cv2.VideoCapture('result/fountain01_den.mp4')
cap.set(cv2.CAP_PROP_FPS, 30)
index = 0
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break
    index += 1

    cv2.imshow('output', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print(index)
cap.release()
cv2.destroyAllWindows()
