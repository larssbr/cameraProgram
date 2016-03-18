# To run from terminal
# python calibrate_undisort.py
import cv2
import numpy as np
import os
import imutils

def resizeImages(self,in_width):
    for i, img in enumerate(self.images):
        self.images[i] = imutils.resize(self.images[i], width=in_width)


def claheAdjustImages(img):
    # --> This method does clahe on lab space to keep the color
    # transform to lab color space and conduct clahe filtering on l channel then merge and transform back

    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe = cv2.createCLAHE(clipLimit=6.0,tileGridSize=(8, 8))
    #self.logger.info("The clahe type {}".format(type(clahe.type)))
    print 'color adjusting image'
    lab_image = cv2.cvtColor(img, cv2.cv.CV_RGB2Lab)
    print 'converted image to Lab space'
    lab_planes = cv2.split(lab_image)
    lab_planes[0] = clahe.apply(lab_planes[0])
    print 'apply clahe to image channel 0 --> L sapce'
    # Merge the the color planes back into an Lab image
    lab_image = cv2.merge(lab_planes, lab_planes[0])
    print 'merge channels back and transform back to rgb space'
    return cv2.cvtColor(lab_image, cv2.cv.CV_Lab2RGB)

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

def fixCalibrationOriginal(folderpath_calib_images, newFolderName):
    for filename in os.listdir(folderpath_calib_images):
        img = cv2.imread(os.path.join(folderpath_calib_images, filename))
        if img is not None:
            # 1 run clahe on the image
            img = claheAdjustImages(img)

            # 2 convert to grayscale
            grey_image = cv2.cvtColor(img, cv2.cv.CV_RGB2GRAY)

            # apply treshold
            # Set threshold and maxValue
            thresh=70
            maxValue=255

            # Threshold to Zero ( type = THRESH_TOZERO )
            #th, THRESH_TOZERO_img = cv2.threshold(grey_image, thresh, maxValue, cv2.THRESH_TOZERO)

            #th, THRESH_TOZERO_INV_img = cv2.threshold(grey_image, thresh, maxValue, cv2.THRESH_BINARY_INV)
            #th, THRESH_TOZERO_INV_img = cv2.threshold(THRESH_TOZERO_INV_img, thresh, maxValue, cv2.THRESH_BINARY_INV)
            th, dst = cv2.threshold(grey_image, thresh, maxValue, cv2.THRESH_BINARY_INV)


            # addWeighted
            blured_image = cv2.GaussianBlur(dst, (5, 5), 105) # //hardcoded filter size, to be tested on 50 mm lens
            wightedGray = cv2.addWeighted(src1=dst, alpha=0.8, src2=grey_image, beta=-0.8, gamma=0.1)

            # # TODO: try resize the image  resize()

            # 3 save image to folder
            # images.append(img)
            cv2.imwrite(os.path.join(newFolderName, filename), wightedGray)

# Finally the function will go through the calbration images and display the undistorted image.
def ImageProcessing(folderpath_calib_images, board_w, board_h, board_dim, image_size):
    # 0 get images
    #folderpath_images = r"C:\MASTER_DATASET\Desembertokt\Mosaic Camera\Calibration Pictures\test_calib_images" # r is there becouse of --> http://stackoverflow.com/questions/7268618/python-issues-with-directories-that-have-special-characters
    #images_list = os.listdir(folderpath_images)

    images_list = load_images_from_folder(folderpath_calib_images)

    nr_of_pics = len(images_list)
    n_boards =  nr_of_pics #2

    # 1 ----- Initializing variables
    objectPoints = []
    imagePoints = []

    npts = np.zeros((n_boards, 1), np.int32)
    intrinsic_matrix = np.zeros((3, 3), np.float32)
    distCoeffs = np.zeros((5, 1), np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

    # 2 ---- prepare object points based on the actual dimensions of the calibration board
    # like (0,0,0), (25,0,0), (50,0,0) ....,(200,125,0)
    # preallocate
    object_points = np.zeros((board_h*board_w, 3), np.float32)
    # fill
    object_points[:, :2] = np.mgrid[0:(board_w*board_dim):board_dim, 0:(board_h*board_dim):board_dim].T.reshape(-1, 2)

    # 3 Loop through the images.  Find checkerboard corners and save the data to imagePoints.
    for i in range(1, n_boards): #range(1, n_boards + 1):

        #Loading images
        print 'Loading... Calibration_Image' # + str(i) + '.jpg'  #'.png'
        # image = cv2.imread('Calibration_Image' + str(i) + '.jpg' )  # '.png')
        #image = cv2.imread(images_list[i])
        image = images_list[i]

        print 'image = images_list[i]'

        # ---> Find chessboard corners
        # needs checkerboard with white bacground arounf them --> http://stackoverflow.com/questions/17993522/opencv-findchessboardcorners-function-is-failing-in-a-apparently-simple-scenar/20187143#20187143
        [found, corners] = cv2.findChessboardCorners(image=image, patternSize=(board_w, board_h), corners=board_w*board_h,
                                                     flags=cv2.cv.CV_CALIB_CB_ADAPTIVE_THRESH + cv2.cv.CV_CALIB_CB_FILTER_QUADS + cv2.cv.CV_CALIB_CB_NORMALIZE_IMAGE)
                                                     #cv2.cv.CV_CALIB_CB_ADAPTIVE_THRESH) #+ cv2.cv.CV_CALIB_CB_NORMALIZE_IMAGE)


        # bool found = findChessboardCorners(image, board_sz, corners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS)

        if found:
            print (found) # TODO: it says findChessboardCorners has not found the chessboard corners
            print ('found chessboard corners')
        else:
            print found
            print "did not find chessboard in image" + str(i)

        if found == True:

            #Add the "true" checkerboard corners
            objectPoints.append(objp)
            # len(objectPoints0 = 0 --> it finds no checkerboard... --> found == FALSE

            #Improve the accuracy of the checkerboard corners found in the image and save them to the imagePoints variable.
            cv2.cornerSubPix(image, corners, (20, 20), (-1, -1), criteria)
            imagePoints.append(corners)

            #Draw chessboard corners
            cv2.drawChessboardCorners(image, (board_w, board_h), corners, found)

            #Show the image with the chessboard corners overlaid.
            cv2.imshow("Corners", image)
            cv2.waitKey(0)

        char = cv2.waitKey(0)

    cv2.destroyWindow("Corners")

    print ''
    print 'Finished processes images.'

    # 4 Calibrate the camera
    print 'Running Calibrations...'
    print(' ')

    # calibrateCamera(pointsmm,points,Size(640,480),cameraMatrix, distCoeffs, rvec, tvec );
    if found == True:
        print ('len(objectPoints) : ' + str(len(objectPoints)))
        print ('objectPoints : ' + objectPoints[0])
        print ('imagePoints : ' + imagePoints)
        print ('image_size : ' + image_size)

        [ret, intrinsic_matrix, distCoeff, rvecs, tvecs] = cv2.calibrateCamera(objectPoints, imagePoints, image_size, None, None) # image.shape[::-1], None, None)

        # 5 Save matrices
        print('Intrinsic Matrix: ')
        print(str(intrinsic_matrix))
        print(' ')
        print('Distortion Coefficients: ')
        print(str(distCoeff))
        print(' ')

        # 6 Save data
        print 'Saving data file...'
        np.savez('calibration_data', distCoeff=distCoeff, intrinsic_matrix=intrinsic_matrix)
        print 'Calibration complete'

        # 7 Calculate the total reprojection error.  The closer to zero the better.
        tot_error = 0
        for i in xrange(len()): # maybe n_boards
            imgpoints2, _ = cv2.projectPoints([i], rvecs[i], tvecs[i], intrinsic_matrix, distCoeff)
            error = cv2.norm(imagePoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            tot_error += error

        print "total reprojection error: ", tot_error/len()

        # return variables to be used by UndistortImages
        return intrinsic_matrix, distCoeff

    else:
        print found
        print "did not find chessboard in image" + str(i)


    cv2.destroyAllWindows()


def UndistortImages(folderpath_undistort_theese_images, intrinsic_matrix, distCoeff):
      # ---->  8 Undistort Images and display them

    images_list = load_images_from_folder(folderpath_undistort_theese_images)
    nr_of_pics = len(images_list)

    for i in range(1, nr_of_pics + 1):
        # 1 Loading images
        print 'Loading... Calibration_Image' + str(i) + '.png'
        image = cv2.imread(images_list(i))

        # 2 Undistort the Image
        undistorted_img = cv2.undistort(image, intrinsic_matrix, distCoeff, None)

        # 3 show the undistorted image
        cv2.imshow('Undisorted Image', undistorted_img)
        cv2.waitKey(0)

        # 4 save the undistorted image

    cv2.destroyAllWindows()

# The program is run from here ###################################################
# Import Information
# folderpath_images = r"C:\MASTER_DATASET\Desembertokt\Mosaic Camera\Calibration Pictures\test_calib_images" # r is there becouse of --> http://stackoverflow.com/questions/7268618/python-issues-with-directories-that-have-special-characters

folderpath_undistort_theese_images = "somewhere"

#images_list = os.listdir(folderpath_images)
#nr_of_pics = len(images_list)

#Input the number of board images to use for calibration (recommended: ~20)
# n_boards =  nr_of_pics #2
#Input the number of squares on the board (width and height)
board_w = 19 # 9
board_h = 15 # 6
#Board dimensions (typically in cm)
board_dim = 80
#Image resolution
image_size = (4008, 2672) # TODO: get the image size directly from the image
'''
# stereo rig : (1360, 1024)
# mono rig = (4008, 2672)
# gopro rig = (1920, 1080)
'''

# runs the method from here

# since the calibration board does not have a white boarder around them
# first "FIX" the calibration images

# save them to newFolderName
newFolderName = "fixed_Calibration_Images"
#folderpath_calib_images = r"calibrationIMG"
folderpath_calib_images = r"gimp_corrected_calibrationIMG"


print("Starting camera calibration....")
print("Step 1: Image fixing")
print("calibration images.")
print(" ")
if not os.path.isdir(newFolderName):
    os.mkdir(newFolderName)
    fixCalibrationOriginal(folderpath_calib_images, newFolderName)
else:
    fixCalibrationOriginal(folderpath_calib_images, newFolderName)
# they are saved as:

print(' ')
print('All the calibration images are fixed.')
print('------------------------------------------------------------------------')

# get the calibration parameters
# from images in folder
print('Step 2: Calibration of fixed images')
print('We will analyze the images and calibrate the camera.')
print(' ')
#folderpath_calib_images = r"fixed_Calibration_Images"
folderpath_calib_images
[intrinsic_matrix, distCoeff] = ImageProcessing(folderpath_calib_images, board_w, board_h, board_dim, image_size)

# undistort images using calibration data : intrinsic_matrix & distCoeff
print('Step 3:  and rectify the original images')
print('we use the intrinsic_matrix & distCoeff to undistort the images.')
print(' ')
#UndistortImages(folderpath_undistort_theese_images, intrinsic_matrix, distCoeff)

# rectify images