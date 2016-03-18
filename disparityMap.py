# python disparityMap.py
import cv2
import numpy as np
from matplotlib import pyplot as plt


# ---->  Next we find the epilines. Epilines corresponding to the points in first image is drawn on second image.
# So mentioning of correct images are important here.
#  We get an array of lines. So we define a new function to draw these lines on the images.

def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
    lines - corresponding epilines '''
    [r, c] = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0,255,3).tolist()) # create a random color
        [x0, y0] = map(int, [0, -r[2]/r[1] ])
        [x1, y1] = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0, y0), (x1,y1), color, 1)

        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
        return img1, img2


# So first we need to find as many possible matches between two images to find the fundamental matrix. For this, we use SIFT descriptors with FLANN based matcher and ratio test.


# WARNING: program needs to run on rectified images

#img1 = cv2.imread('calibrationIMG/calibration_image1.jpg', 0)  #queryimage # left image
#img2 = cv2.imread('calibrationIMG/calibration_image2.jpg', 0) #trainimage # right image

img1 = cv2.imread('images/calib_images/left01.jpg', 0)
img2 = cv2.imread('images/calib_images/right01.jpg', 0)


# find the keypoints and descriptors with SIFT
sift = cv2.SIFT()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1, des2, k=2)

good = []
pts1 = []
pts2 = []

# --->  Get the best matches from both the images.
# ratio test as per Lowe's paper
for i, (m, n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)
# Now we have the list of best matches from both the images.

# pts1 = np.int32(pts1)
# pts2 = np.int32(pts2)

pts1 = np.float32(pts1) # float32(pts1)
pts2 = np.float32(pts2) # float32(pts2)
print(pts1 )
#print('pts2 : ' + pts2 )

# --> Let's find the Fundamental Matrix.
[F, mask] = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)


# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

# Rectify undistoreted
[height, width] = tuple(img1.shape[1::-1])
imageSize = (height, width)
rectification_homography1, rectification_homography2 = cv2.cv.StereoRectifyUncalibrated(pts1, pts2, F, imageSize,  threshold=5)


# warpperspective
warpImage1 = cv2.warpPerspective(img1, rectification_homography1, imageSize)
warpImage2 = cv2.warpPerspective(img2, rectification_homography2, imageSize);

# ------> Now we find the epilines in both the images and draw them.

 # Find epilines corresponding to points in right image (second image) and
 # drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
#  The format of point matrix is unsupported in function cvComputeCorrespondEpilines
# --> needs to run program on rectified images
lines1 = lines1.reshape(-1,3)
[img5, img6] = drawlines(img1, img2, lines1, pts1, pts2)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
[img3, img4] = drawlines(img2, img1, lines2, pts2, pts1)

plt.subplot(121), plt.imshow(img5)
plt.subplot(122), plt.imshow(img3)
plt.show()

'''
Exercises

One important topic is the forward movement of camera. Then epipoles will be seen at the same locations in both with epilines emerging from a fixed point. See this discussion.
Fundamental Matrix estimation is sensitive to quality of matches, outliers etc. It becomes worse when all selected matches lie on the same plane. Check this discussion.
'''


'''
imgL = cv2.imread('calibrationIMG/calibration_image1.jpg', 0)
imgR = cv2.imread('calibrationIMG/calibration_image2.jpg', 0)

# Create disparity images
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL, imgR)
plt.imshow(disparity, 'gray')
plt.show()
'''