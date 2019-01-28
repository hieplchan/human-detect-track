from optical_flow_tracking.utils.params import *

""" VARIABLE DEFINITION """
# Tracking point
points_origin = []

""" COMMON FUNCTION """
# Mouse function
def mouse_select(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        points_origin.append([x, y])
        print(str(x) + ":" + str(y))
        for point in points_origin:
            cv2.circle(frame, (point[0], point[1]), 5, (0, 0, 255), -1)
            cv2.imshow("CCTV", frame)

""" MAIN FUNCTION """
# Mouse function
if __name__ == '__main__':
    # Set up
    cap = cv2.VideoCapture(VIDEO_PATH + VIDEO_NAME)
    cv2.namedWindow("CCTV")
    cv2.setMouseCallback("CCTV", mouse_select)

    # Point selection
    ret, frame = cap.read()
    cv2.imshow("CCTV", frame)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    while(True):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            points_origin = np.array(points_origin, dtype=np.float32)
            points_old = points_origin.copy()
            print(points_origin.shape)
            break

    # Lucas Kanade Tracking
    while(True):
        logger.info(time.process_time())
        ret, frame = cap.read()
        old_gray_frame = gray_frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        points_new, status, error = cv2.calcOpticalFlowPyrLK(old_gray_frame, gray_frame, points_old, None, **lk_params)
        points_old = points_new.copy()

        print(points_old)

        # for point in points_new:
        #     x, y = point.ravel()
        #     cv2.circle(frame, (x,y) ,5, (0, 0, 255), -1)
        #     cv2.imshow("CCTV", frame)

        cv2.imshow("CCTV", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
