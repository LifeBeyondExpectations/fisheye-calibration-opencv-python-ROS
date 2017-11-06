#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import cv2

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import rospy
import roslib

import pickle
from tempfile import TemporaryFile

import glob
import os


#flag
flag_subscribe_new_image_not_load_old_image = 1
flag_publish_calibrated_image = 1


flag_load_calibrated_result = 1
flag_load_detected_result = 0
flag_save_image_onlyWhichDetectCheckeboard = 0
flag_1_show_image_2_homography_3_distorted = 1

flag_print = 0


#parameter
ROS_TOPIC = 'jaesung_lens_camera/image_color'
path_image_database = "video_francois_lens/*.png"
save_video_to_the_path = "video_francois_lens/"
nameOf_pickle_Checkerboard_Detection = 'result/detection_result_francois_171021_1620_delete_image'
nameof_pickel_calibrated_result = "calib_result_JS_fisheye.pickle"
fileList = []
num_of_image_in_database = 1000
count = 0
mybalance = 0

class calibration:
    # you must check the checkboard which you would like to use for the calibration
    # if (9, 7) it means there are 10(=9+1) pactches in row, 8(=7+1) patches in column
    CHECKERBOARD = (9, 7)  # (8, 8)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = []

    flag_calibratedResult_save = 1
    flag_get_undistort_param = 1
    flag_first_didHomography = 1
    height = 0
    width = 0

    def __init__(self):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((1, self.CHECKERBOARD[0] * self.CHECKERBOARD[1], 3), np.float32)
        self.objp[0, :, :2] = np.mgrid[0:self.CHECKERBOARD[0], 0:self.CHECKERBOARD[1]].T.reshape(-1, 2)

    def detectCheckerBoard(self, frame):
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # https://docs.opencv.org/trunk/d9/d0c/group__calib3d.html#ga93efa9b0aa890de240ca32b11253dd4a
        ret, corners = cv2.findChessboardCorners(image=gray,
                                                 patternSize=self.CHECKERBOARD,
                                                 flags=(cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_NORMALIZE_IMAGE))

        # If found, add object points, image points (after refining them)
        if ret == True:
            print(">> detected the checkerboard")

            # Save images if wanted
            if flag_subscribe_new_image_not_load_old_image == 1 and flag_save_image_onlyWhichDetectCheckeboard == 1:
                cv2.imwrite(save_video_to_the_path + (str((count + 0)) + '.png'), frame)

            # https://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html#void cornerSubPix(InputArray image, InputOutputArray corners, Size winSize, Size zeroZone, TermCriteria criteria)
            corners2 = cv2.cornerSubPix(image=gray,
                                        corners=corners,
                                        winSize=(11, 11),
                                        zeroZone=(-1, -1),
                                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

            self.imgpoints.append(corners2)
            self.objpoints.append(self.objp)

            # Draw and display the corners
            frame = cv2.drawChessboardCorners(image=frame, patternSize=self.CHECKERBOARD, corners=corners2, patternWasFound=ret)

        # Display the resulting frame
        cv2.namedWindow('checkerboard detected frame')
        cv2.imshow('checkerboard detected frame', frame)
        cv2.waitKey(1)

    def saveVarInDetection(self):
        with open(nameOf_pickle_Checkerboard_Detection, 'w') as f:
            pickle.dump([self.objpoints, self.imgpoints, self.width, self.height], f)

    def loadVarInDetection(self):
        with open(nameOf_pickle_Checkerboard_Detection) as f:
            self.objpoints, self.imgpoints, self.width, self.height = pickle.load(f)

            global flag_print
            if flag_print == 1:
                # check the result
                tmp = np.array(self.objpoints)
                print("shape of objpoints is ", tmp.shape)
    
                tmp = np.array(self.imgpoints)
                print("shape of imgpoints is ", tmp.shape)
    
                print("width is ", self.width, "height is ", self.height)

    def startCalibration(self):        
        
        self.camera_matrix = np.zeros((3, 3))
        self.distCoeffs = np.zeros((4, 1))

        N_OK = len(self.objpoints)
        self.rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        self.tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        self.ret, _, _, _, _ = cv2.fisheye.calibrate(
            objectPoints=self.objpoints,
            imagePoints=self.imgpoints,
            image_size=(self.width, self.height),
            K=self.camera_matrix,
            D=self.distCoeffs,
            rvecs=None,#self.rvecs,
            tvecs=None,#self.tvecs,
            flags=(cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW), # +cv2.fisheye.CALIB_FIX_PRINCIPAL_POINT cv2.fisheye.CALIB_CHECK_COND
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-6))

        global flag_print
        if flag_print == 1:
            # check the result of calibration
            print('camera matrix is ')
            print(self.camera_matrix)
            # print('new camera Matrix is ')
            # print(self.new_camera_matrix)
            # print('self.rvecs is ')
            # print(self.rvecs)
            # print('self.tvecs is ')
            # print(self.tvecs)
            print('distort Coeffs is ')
            print(self.distCoeffs)
            print('RMS re-projection error is ', self.ret)            

        print("calibration complete")

    def saveVarAfterCalibration(self):
        with open(nameof_pickel_calibrated_result, 'w') as f:
            pickle.dump([self.camera_matrix, self.distCoeffs, self.new_camera_matrix, self.width, self.height], f)
            # self.camera_matrix, self.distCoeffs, self.new_camera_matrix, self.width, self.height, self.roi, self.map1, self.map2
            
    def loadVarAfterCalibration(self):
        with open(nameof_pickel_calibrated_result) as f:
            self.camera_matrix, self.distCoeffs, self.new_camera_matrix, self.width, self.height, self.roi, self.map1, self.map2 = pickle.load(f)
    
        global flag_print
        if flag_print == 1:
            print('camera matrix is ')
            print(self.camera_matrix)
            print('new camera Matrix is ')
            print(self.new_camera_matrix)
            print('distort Coeffs is ')
            print(self.distCoeffs)
            print('width, height is ', self.width, self.height)

    def undistort_imShow(self, frame):
        dim0 = frame.shape[:2][::-1] #(width, height)

        frame_undistorted = cv2.fisheye.undistortImage(frame, self.camera_matrix, self.distCoeffs, Knew=self.camera_matrix, new_size=dim0)

        global flag_1_show_image_2_homography_3_distorted

        if flag_1_show_image_2_homography_3_distorted == 1:
            cv2.namedWindow('JS calibrated resuls')
            cv2.imshow('JS calibrated resuls', frame_undistorted)

        elif flag_1_show_image_2_homography_3_distorted == 2:
            #show top view image based on homography
            tmp = self.wrapper_homography(frame_undistorted)

        elif flag_1_show_image_2_homography_3_distorted == 3:
            cv2.namedWindow('JS distorted(original) frames')
            cv2.imshow('JS distorted(original) frames', frame)

        else:
            print('wrong flag_1_show_image_2_homography_3_distorted')

        cv2.waitKey(1)
        return frame_undistorted

    def wrapper_homography(self, frame_JS):

        if self.flag_first_didHomography == 1:
            # Jaesung
            srcPoints_JS = np.array([[63, 133], [149, 122], [250, 113], [374, 105], [520, 98],
                                                                        [372, 126],
                                                                        [372, 153],
                                                [133, 180], [239, 177], [371, 172], [527, 173],

                                                #[125, 241], [239, 244], [367, 250], [531, 256],
                                     [19, 313], [108, 319], [222, 331], [366, 343], [537, 352],
                                                                        [365, 375],
                                                                        [364, 405],
                                     [9, 379], [101, 395], [218, 412], [362, 436], [537, 453],
                                     [4, 445], [95, 467]])



            dstPoints_JS = np.array([[0,0],    [0,  50],  [0,100],    [0, 150],    [0, 200],
                                                                      [16, 150],
                                                                      [34, 150],
                                               [50, 50],  [50, 100],  [50, 150],  [50, 200],

                                               #[100, 50], [100, 100], [100, 150], [100, 200],
                                     [150, 0], [150, 50], [150, 100], [150, 150], [150, 200],
                                                                      [166, 150],
                                                                      [184, 150],
                                     [200, 0], [200, 50], [200, 100], [200, 150], [200, 200],
                                     [250, 0], [250, 50]])



            srcPoints_JS, dstPoints_JS = np.array(srcPoints_JS), np.array(dstPoints_JS)
            self.homography_JS, mask = cv2.findHomography(srcPoints=srcPoints_JS, dstPoints=dstPoints_JS, method=cv2.RANSAC)

            self.flag_first_didHomography = 0

        frame_homography_JS = cv2.warpPerspective(frame_JS, self.homography_JS, (250, 250))

        # cv2.namedWindow('Undistorted frame of JS fisheye camera')
        # cv2.imshow('Undistorted frame of JS fisheye camera', frame_JS)#cv2.resize(frame_JS, (0,0), fx=2, fy=2))

        cv2.namedWindow('Transformation of undistorted frame of JS fisheye camera using homography')
        cv2.imshow('Transformation of undistorted frame of JS fisheye camera using homography', cv2.resize(frame_homography_JS, (0,0), fx=2, fy=2))

        cv2.waitKey(1)

        return frame_homography_JS

class dataLoadType:

    singleImage_inst = []
    calibrate_inst = []
    flag_fisrt_didLoadVarDetection = 1
    flag_first_didLoadVarCalibration = 1

    def __init__(self, singleImage_inst, calibrate_inst):
        self.singleImage_inst = singleImage_inst
        self.calibrate_inst = calibrate_inst
        self.bridge = CvBridge()

    def subscribeImage(self):
        print('start to subscribe image')
        #rospy.init_node('dataLoadType', anonymous=True)
        self.rospySubImg = rospy.Subscriber(ROS_TOPIC, Image, self.callback)
        #automatically go to the callback function : self.callback()

    def stopSubscribing(self):
        print('stop subscribing image')
        #self.image_sub.shutdown()
        self.rospySubImg.unregister()

    def loadImageInFiles(self):
        global fileList
        fileList = glob.glob(path_image_database)
        print('path_image_database is ', path_image_database)

        global num_of_image_in_database
        num_of_image_in_database = len(fileList)
        print('what is fileList', fileList, '\n')

        global count
        for i in fileList:
            count = count + 1
            print('The ', count, '-th image is under processing')
            self.singleImage_inst.saveImage(cv2.imread(i))

            self.wrapper()

    def publishImage(self):
        # http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscriber%28python%29
        print('start to publish JS image')
        self.rospyPubImg = rospy.Publisher('calibration_JS_fisheye/image_calibrated', Image, queue_size=10)
        rospy.Rate(10)  # 10Hz


    def callback(self, data):
        global count
        try:
            # parse message into image
            # bgr8: CV_8UC3, color image with blue-green-red color order and 8bit
            self.singleImage_inst.saveImage(self.bridge.imgmsg_to_cv2(data, "bgr8"))
            count = count + 1

            # if you want to work asynchronously, edit the lines below
            self.wrapper()

        except CvBridgeError as e:
            print(e)

    def wrapper(self):
        global count, num_of_image_in_database
        global flag_subscribe_new_image_not_load_old_image, flag_load_calibrated_result

        self.calibrate_inst.height = self.singleImage_inst.height
        self.calibrate_inst.width = self.singleImage_inst.width

        if flag_load_detected_result == 0 and flag_load_calibrated_result == 0:
            if count < num_of_image_in_database:
                self.calibrate_inst.detectCheckerBoard(self.singleImage_inst.imgData)
            elif count == num_of_image_in_database:
                self.calibrate_inst.detectCheckerBoard(self.singleImage_inst.imgData)
                self.calibrate_inst.saveVarInDetection()
                self.calibrate_inst.startCalibration()
                self.calibrate_inst.saveVarAfterCalibration()
            else:
                self.singleImage_inst.imgCalibrated = self.calibrate_inst.undistort_imShow(self.singleImage_inst.imgData)

        elif flag_load_detected_result == 1 and flag_load_calibrated_result == 0:
            # load Detection results, start calibratin and show undistorted image
            if self.flag_fisrt_didLoadVarDetection == 1:
                self.calibrate_inst.loadVarInDetection()
                self.calibrate_inst.startCalibration()
                self.calibrate_inst.saveVarAfterCalibration()
                # do not come here again
                self.flag_fisrt_didLoadVarDetection = 0

            self.singleImage_inst.imgCalibrated = self.calibrate_inst.undistort_imShow(self.singleImage_inst.imgData)

        elif flag_load_calibrated_result == 1:
            # load Calibration results and show undistorted image
            if self.flag_first_didLoadVarCalibration == 1:
                self.calibrate_inst.loadVarAfterCalibration()
                # do not come here again
                self.flag_first_didLoadVarCalibration = 0

            self.singleImage_inst.imgCalibrated = self.calibrate_inst.undistort_imShow(self.singleImage_inst.imgData)

        else:
            print("fucking error for count = ", count)

        if flag_publish_calibrated_image == 1:
            try:
                self.rospyPubImg.publish(self.bridge.cv2_to_imgmsg(self.singleImage_inst.imgCalibrated, "bgr8"))
            except CvBridgeError as e:
                print(e)

class singleImageData:
    height = 0
    width = 0
    imgData = None
    imgCalibrated = None

    def saveImage(self, img):
        self.imgData = img
        self.height, self.width = self.imgData.shape[:2]

if __name__ == "__main__":

    print("check the version opencv.")
    print(cv2.__version__)

    #global count
    count = 0

    singleImage_inst = singleImageData()
    calibrate_inst = calibration()
    dataLoadType_inst = dataLoadType(singleImage_inst, calibrate_inst)

    #global flag_subscribe_new_image_not_load_old_image
    try:
       if flag_subscribe_new_image_not_load_old_image == 1:
            # One python file for one init_node
            rospy.init_node('calibration_JS_fisheye', anonymous=True)
            dataLoadType_inst.subscribeImage()

            if flag_publish_calibrated_image == 1:
                dataLoadType_inst.publishImage()

            rospy.spin()

       elif flag_subscribe_new_image_not_load_old_image == 0:
           dataLoadType_inst.loadImageInFiles()

           if flag_publish_calibrated_image == 1:
                dataLoadType_inst.publishImage()

    except KeyboardInterrupt:
        print("Shutting down")

    cv2.destroyAllWindows()