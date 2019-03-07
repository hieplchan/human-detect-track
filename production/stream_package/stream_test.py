import cv2
import time

#Video load for test
cap = cv2.VideoCapture('/media/hiep/DATA/Work_space/Tracking_CCTV/CCTV_Data/Video/4.mp4')

if __name__ == "__main__":
    while (True):
        res, draw_image = cap.read()
        if (res != 1):
            break
        else:
            cv2.imshow('Frame', draw_image)
            cv2.waitKey(1)
