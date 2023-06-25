import numpy as np
import cv2
from matplotlib import pyplot as plt

def watershed(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Noise removal
    kernel = np.ones((2,2),np.uint8)
    closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 2)
    sure_bg = cv2.dilate(closing,kernel,iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(sure_bg,cv2.DIST_L2,3)
    # Threshold
    ret, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    markers = cv2.watershed(img,markers)
    img[markers == -1] = [255,0,0]

    # Use matplotlib because cv2 can't work 
    # I don't know why either ;____;
    plt.subplot(1, 2, 1),plt.imshow(unknown, 'gray')
    plt.title("Unknown"), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2),plt.imshow(img)
    plt.title("Watershed"), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()
