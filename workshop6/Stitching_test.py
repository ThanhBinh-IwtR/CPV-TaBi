import cv2
import numpy as np
import imutils
import imageio

#read img and grayscale
ri_img = cv2.imread('IMG/6.jpg')
le_img = cv2.imread('IMG/5.jpg')
src_gray = cv2.cvtColor(ri_img, cv2.COLOR_RGB2GRAY)
tar_gray = cv2.cvtColor(le_img, cv2.COLOR_RGB2GRAY)

height = max(ri_img.shape[0], le_img.shape[0])
width1 = int(ri_img.shape[1] * height / ri_img.shape[0])
width2 = int(le_img.shape[1] * height / le_img.shape[0])
image1 = cv2.resize(ri_img, (width1, height))
image2 = cv2.resize(le_img, (width2, height))

combined_image = np.concatenate((image1, image2), axis=1)

cv2.imshow("Input img", combined_image)

#use sift to detect
SIFT_detector = cv2.SIFT_create()
kp1, des1 = SIFT_detector.detectAndCompute(src_gray, None)
kp2, des2 = SIFT_detector.detectAndCompute(tar_gray, None)

## Match keypoint
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

## Bruce Force KNN trả về list k ứng viên cho mỗi keypoint.
rawMatches = bf.knnMatch(des1, des2, 2)
matches = []

for m,n in rawMatches:
    # giữ lại các cặp keypoint sao cho với kp1, khoảng cách giữa kp1 với ứng viên 1 nhỏ hơn nhiều so với khoảng cách giữa kp1 và ứng viên 2
    if m.distance < n.distance * 0.75:
        matches.append(m)

# do có cả nghìn match keypoint, ta chỉ lấy tầm 100 -> 200 cặp tốt nhất để tốc độ xử lí nhanh hơn
matches = sorted(matches, key=lambda x: x.distance, reverse=True)
matches = matches[:200]

img3 = cv2.drawMatches(ri_img, kp1, le_img, kp2, matches, None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# img3 = cv2.resize(img3, (width2, height//2))
cv2.namedWindow('drawmatchers', cv2.WINDOW_AUTOSIZE)
cv2.imshow('drawmatchers', img3)

## Nhìn vào hình dưới đây, ta thấy các cặp Keypoint giữa 2 ảnh đã được match khá chính xác, số điểm nhiễu không quá nhiều
kp1 = np.float32([kp.pt for kp in kp1])
kp2 = np.float32([kp.pt for kp in kp2])
pts1 = np.float32([kp1[m.queryIdx] for m in matches])
pts2 = np.float32([kp2[m.trainIdx] for m in matches])

# estimate the homography between the sets of points
(H, status) = cv2.findHomography(pts1, pts2, cv2.RANSAC)
print(H)

#stitching img
h1, w1 = ri_img.shape[:2]
h2, w2 = le_img.shape[:2]
result = cv2.warpPerspective(ri_img, H, (w1+w2, h1))
result[0:h2, 0:w2] = le_img
cv2.namedWindow('result', cv2.WINDOW_AUTOSIZE)
cv2.imshow('result', result)
cv2.waitKey(0)

