import cv2
import numpy as np

from optical_flow_tracking.utils.params import *

if __name__ == '__main__':
    cap = cv2.VideoCapture(VIDEO_PATH + VIDEO_NAME)
    while(True):
        logger.info(time.process_time())
        ret, frame = cap.read()
        cv2.imshow("CCTV", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
