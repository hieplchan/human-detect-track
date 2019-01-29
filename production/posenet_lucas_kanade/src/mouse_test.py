import cv2
import numpy as np

# mouse callback function
def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.rectangle(mask,(x-4,y-4),(x+4,y+4),(255,0,0),-1)

# Create a black image, a window and bind the function to window
img = np.zeros((512,512,3), np.uint8)
mask = np.zeros((512,512,1), np.uint8)
print(img.shape)
print(mask.shape)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

while(1):
    cv2.imshow('image',img)
    cv2.imshow('mask',mask)
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()
