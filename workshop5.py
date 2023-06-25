import cv2
import numpy as np

MAX_FEATURES = 1000
GOOD_MATCH_PERCENT = 0.5

im1 = cv2.imread('IMG/1.jpg')
im2 = cv2.imread('IMG/test3.jpg')

# Convert images to grayscale
im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
height, width = im1Gray.shape

# Detect ORB features and compute descriptors.
orb = cv2.ORB_create(MAX_FEATURES)
keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

# Match features.
# matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
# matches = matcher.match(descriptors1, descriptors2, None)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(descriptors1, descriptors2)

# Remove not so good matches
numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
matches = matches[:numGoodMatches]

# Draw top matches
imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
imMatches = cv2.resize(imMatches, (width , height ))
cv2.imwrite("matches.jpg", imMatches)
cv2.namedWindow('Image Match', cv2.WINDOW_AUTOSIZE)
cv2.imshow('Image Match', imMatches)

#draw keypoint
imgKp_Ref = cv2.drawKeypoints(im1, keypoints2, 0, (0, 0, 222), None)
imgKp_Ref = cv2.resize(imgKp_Ref, (width , height ))
cv2.namedWindow('Key Points', cv2.WINDOW_AUTOSIZE)
cv2.imshow('Key Points', imgKp_Ref)
# cv2.waitKey(0)

# Extract location of good matches
points1 = np.zeros((len(matches), 2), dtype=np.float32)
points2 = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt

# Find homography
h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
# Use homography
aligned_img = cv2.warpPerspective(im1, h, (width, height))
aligned_img = cv2.resize(aligned_img, (width // 4, height // 4))
print("Estimated homography : \n", h)
# cv2.namedWindow('Output Image', cv2.WINDOW_AUTOSIZE)
# cv2.imshow('Output Image', aligned_img)
cv2.waitKey(0)