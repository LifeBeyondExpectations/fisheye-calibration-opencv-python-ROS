#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import cv2

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import message_filters

import rospy
import roslib

import pickle

import glob
import os

import numpy as np
import cv2
import sys

import time


#https://github.com/kushalvyas/Python-Multiple-Image-Stitching/blob/master/code/pano.py
import cv2
import numpy as np

#flag
flag_subscribe_new_image_not_load_old_image = 1
flag_publish_calibrated_image = 1

flag_load_calibrated_result = 1
flag_load_detected_result = 0
flag_fisheye_calibrate = 1
flag_save_image_onlyWhichDetectCheckeboard = 0
flag_show_image = 1
flag_cctv_run = 1

#paramter
fileList = []
num_of_image_in_database = 1000
count = 0
mybalance = 0



#ROS_TOPIC = 'francois_lens_camera/image_color'
path_image_database = 'stereo/*.png'
save_video_to_the_path = 'stereo/'
nameOf_pickle_Checkerboard_Detection = "result/detect_result_test_seethrough_line_171024.pickle"
nameof_pickel_calibrated_result = "result/calib_result_test_seethrough_line_171024.pickle"

class matchers:
    def __init__(self):
        self.surf = cv2.xfeatures2d.SURF_create()
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=0, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
    def match(self, i1, i2, direction=None):
        imageSet1 = self.getSURFFeatures(i1)
        imageSet2 = self.getSURFFeatures(i2)
        print("Direction : ", direction)
        matches = self.flann.knnMatch(imageSet2['des'],imageSet1['des'],k=2)
        good = []
        for i , (m, n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                good.append((m.trainIdx, m.queryIdx))

        if len(good) > 4:
            pointsCurrent = imageSet2['kp']
            pointsPrevious = imageSet1['kp']

            matchedPointsCurrent = np.float32(
                [pointsCurrent[i].pt for (__, i) in good]
            )
            matchedPointsPrev = np.float32(
                [pointsPrevious[i].pt for (i, __) in good]
            )

            H, s = cv2.findHomography(matchedPointsCurrent, matchedPointsPrev, cv2.RANSAC, 4)
            return H
        return None

    def getSURFFeatures(self, im):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        kp, des = self.surf.detectAndCompute(gray, None)
        return {'kp':kp, 'des':des}

# https://github.com/kushalvyas/Python-Multiple-Image-Stitching/blob/master/code/pano.py
class Stitch:
    images = [[],[]]
    def __init__(self, leftImage, rightImage):
        self.images[0] = cv2.resize(leftImage, (480, 320), cv2.INTER_LINEAR)
        self.images[1] = cv2.resize(rightImage, (480, 320), cv2.INTER_LINEAR)
        #  = [cv2.resize(cv2.imread(each), (480, 320))b for each in (leftImage, rightImage)]
        self.count = 2#len(self.images)
        #print('what is self.count ', self.count)
        self.left_list, self.right_list, self.center_im = [], [], None
        self.matcher_obj = matchers()
        self.prepare_lists()

    def prepare_lists(self):
        print("Number of images : %d" % self.count)

        self.centerIdx = self.count / 2
        print("Center index image : %d" % self.centerIdx)

        self.center_im = self.images[self.centerIdx]
        for i in range(self.count):
            if (i <= self.centerIdx):
                self.left_list.append(self.images[i])
            else:
                self.right_list.append(self.images[i])
        print("Image lists prepared")


    def leftshift(self, Homography):
        # self.left_list = reversed(self.left_list)
        a = self.left_list[0]
        for b in self.left_list[1:]:
            H = Homography
            # H = self.matcher_obj.match(a, b, 'left')
            print("Homography is : ", H)

            xh = np.linalg.inv(H)
            print("Inverse Homography :", xh)

            ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]));
            ds = ds / ds[-1]
            print("final ds=>", ds)

            f1 = np.dot(xh, np.array([0, 0, 1]))
            f1 = f1 / f1[-1]
            xh[0][-1] += abs(f1[0])
            xh[1][-1] += abs(f1[1])
            ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]))
            offsety = abs(int(f1[1]))
            offsetx = abs(int(f1[0]))
            dsize = (int(ds[0]) + offsetx, int(ds[1]) + offsety)
            print("image dsize =>", dsize)

            tmp = cv2.warpPerspective(a, xh, dsize)
            # cv2.imshow("warped", tmp)
            # cv2.waitKey()
            tmp[offsety:b.shape[0] + offsety, offsetx:b.shape[1] + offsetx] = b
            a = tmp

        self.leftImage = tmp

    def rightshift(self, Homography):
        for each in self.right_list:
            #H = self.matcher_obj.match(self.leftImage, each, 'right')
            H = Homography
            print("Homography :", H)

            txyz = np.dot(H, np.array([each.shape[1], each.shape[0], 1]))
            txyz = txyz / txyz[-1]
            dsize = (int(txyz[0]) + self.leftImage.shape[1], int(txyz[1]) + self.leftImage.shape[0])
            tmp = cv2.warpPerspective(each, H, dsize)
            cv2.imshow("tp", tmp)
            cv2.waitKey()
            # tmp[:self.leftImage.shape[0], :self.leftImage.shape[1]]=self.leftImage
            tmp = self.mix_and_match(self.leftImage, tmp)
            print("tmp shape", tmp.shape, "self.leftimage shape=", self.leftImage.shape)
            self.leftImage = tmp
            # self.showImage('left')

    def mix_and_match(self, leftImage, warpedImage):
        i1y, i1x = leftImage.shape[:2]
        i2y, i2x = warpedImage.shape[:2]
        print(leftImage[-1, -1])


        t = time.time()
        black_l = np.where(leftImage == np.array([0, 0, 0]))
        black_wi = np.where(warpedImage == np.array([0, 0, 0]))
        print(time.time() - t)

        print(black_l[-1])


        for i in range(0, i1x):
            for j in range(0, i1y):
                try:
                    if (np.array_equal(leftImage[j, i], np.array([0, 0, 0])) and np.array_equal(warpedImage[j, i], np.array([0, 0, 0]))):
                        # print "BLACK"
                        # instead of just putting it with black,
                        # take average of all nearby values and avg it.
                        warpedImage[j, i] = [0, 0, 0]
                    else:
                        if (np.array_equal(warpedImage[j, i], [0, 0, 0])):
                            # print "PIXEL"
                            warpedImage[j, i] = leftImage[j, i]
                        else:
                            if not np.array_equal(leftImage[j, i], [0, 0, 0]):
                                bw, gw, rw = warpedImage[j, i]
                                bl, gl, rl = leftImage[j, i]
                                # b = (bl+bw)/2
                                # g = (gl+gw)/2
                                # r = (rl+rw)/2
                                warpedImage[j, i] = [bl, gl, rl]
                except:
                    pass
        # cv2.imshow("waRPED mix", warpedImage)
        # cv2.waitKey()
        return warpedImage

    def trim_left(self):
        pass

    def showImage(self, string=None):
        if string == 'left':
            cv2.imshow("left image", self.leftImage)
        # cv2.imshow("left image", cv2.resize(self.leftImage, (400,400)))
        elif string == "right":
            cv2.imshow("right Image", self.rightImage)
        cv2.waitKey()

class calibration:
    # you must check the checkboard which you would like to use for the calibration
    # if (9, 7) it means there are 10(=9+1) pactches in row, 8(=7+1) patches in column
    CHECKERBOARD = (9, 7) # (7, 6) #(9, 7)  # (8, 8)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints_JS = []  # 2d points in image plane.
    imgpoints_Francois = []  # 2d points in image plane.
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)

    objp = []


    flag_first_width_height_save_yet = 1
    flag_calibratedResult_save = 1
    flag_get_undistort_param = 1
    height = 0
    width = 0

    def __init__(self):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)

        # This is the original one
        # self.objp = np.zeros((1, self.CHECKERBOARD[0] * self.CHECKERBOARD[1], 3), np.float64)
        # self.objp[0, :, :2] = np.mgrid[0:self.CHECKERBOARD[0], 0:self.CHECKERBOARD[1]].T.reshape(-1, 2)

        self.objp = np.zeros(( self.CHECKERBOARD[0] * self.CHECKERBOARD[1], 1, 3), np.float64)
        self.objp[ :, 0, :2] = np.mgrid[0:self.CHECKERBOARD[0], 0:self.CHECKERBOARD[1]].T.reshape(-1, 2)



        # self.opts_loc = np.zeros((self.CHECKERBOARD[1] * self.CHECKERBOARD[0], 1, 3))
        # for j in range(self.CHECKERBOARD[1] * self.CHECKERBOARD[0]):
        #     self.opts_loc[j, 0, 0] = (j / self.CHECKERBOARD[1])
        #     self.opts_loc[j, 0, 1] = (j % self.CHECKERBOARD[0])
        #     self.opts_loc[j, 0, 2] = 0
        # self.objp.append(self.opts_loc)

        # print('self.objp is')
        # print(self.objp)

    def detectSteroCheckerBoard(self, frame_JS, frame_Francois):

        if self.flag_first_width_height_save_yet == 1:
            self.width_train_JS, self.height_train_JS = frame_JS.shape[:2][::-1]
            self.width_train_Francois, self.height_train_Francois = frame_Francois.shape[:2][::-1]
            #print(self.width_train_JS, self.height_train_JS, self.width_train_Francois, self.height_train_Francois)
            self.flag_first_width_height_save_yet = 0


        # Our operations on the frame come here
        gray_JS = cv2.cvtColor(frame_JS, cv2.COLOR_BGR2GRAY)

        # https://docs.opencv.org/trunk/d9/d0c/group__calib3d.html#ga93efa9b0aa890de240ca32b11253dd4a
        ret_JS, corners_JS = cv2.findChessboardCorners(image=gray_JS,
                                                 patternSize=self.CHECKERBOARD,
                                                 flags=(cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE))

        # If found, add object points, image points (after refining them)
        if ret_JS == True:
            # print('Jaesung detected')

            # https://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html#void cornerSubPix(InputArray image, InputOutputArray corners, Size winSize, Size zeroZone, TermCriteria criteria)
            corners2_JS = cv2.cornerSubPix(image=gray_JS,
                                        corners=corners_JS,
                                        winSize=(11, 11),
                                        zeroZone=(-1, -1),
                                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

            gray_Francois = cv2.cvtColor(frame_Francois, cv2.COLOR_BGR2GRAY)
            ret_Francois, corners_Francois = cv2.findChessboardCorners(image=gray_Francois,
                                                           patternSize=self.CHECKERBOARD,
                                                           flags=(cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE))

            # Draw and display the corners
            frame_JS_copy = frame_JS.copy()
            cv2.drawChessboardCorners(image=frame_JS, patternSize=self.CHECKERBOARD, corners=corners2_JS, patternWasFound=ret_JS)

            if ret_Francois == True:
                print('Francois detected')

                corners2_Francois = cv2.cornerSubPix(image=gray_Francois,
                                               corners=corners_Francois,
                                               winSize=(11, 11),
                                               zeroZone=(-1, -1),
                                               criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

                frame_Francois_copy = frame_Francois.copy()
                # Draw and display the corners
                cv2.drawChessboardCorners(image=frame_Francois, patternSize=self.CHECKERBOARD, corners=corners2_Francois, patternWasFound=ret_Francois)


                # self.imgpoints_JS.append(corners2_JS)
                # self.imgpoints_Francois.append(corners2_Francois)
                self.imgpoints_JS.append(np.array(corners2_JS))
                self.imgpoints_Francois.append(np.array(corners2_Francois))
                self.objpoints.append(np.array(self.objp))

                if flag_save_image_onlyWhichDetectCheckeboard == 1 and flag_subscribe_new_image_not_load_old_image == 1:
                    cv2.imwrite(save_video_to_the_path + (str((count + 10000)) + '.png'), frame_JS_copy)
                    cv2.imwrite(save_video_to_the_path + (str((count + 10001)) + '.png'), frame_Francois_copy)


        # Display the resulting frame
        # cv2.namedWindow('Detection display')
        # cv2.imshow('Detection display', np.concatenate((cv2.resize(frame_JS, (720, 480), cv2.INTER_LINEAR), cv2.resize(frame_Francois, (720, 480), cv2.INTER_LINEAR)), axis=1))
        # cv2.waitKey(1)

    def saveVarInDetection(self):
        with open(nameOf_pickle_Checkerboard_Detection, 'w') as f:
            pickle.dump([self.objpoints, self.imgpoints_JS, self.imgpoints_Francois, self.width_train_JS, self.height_train_JS, self.width_train_Francois, self.height_train_Francois], f)
        print('saveVarInDetection Complete')

    def loadVarInDetection(self):
        with open(nameOf_pickle_Checkerboard_Detection) as f:
            self.objpoints, self.imgpoints_JS, self.imgpoints_Francois, self.width_train_JS, self.height_train_JS, self.width_train_Francois, self.height_train_Francois = pickle.load(f)

            # check the result
            tmp = np.array(self.imgpoints_Francois)
            self.imgpoints_Francois = np.array(self.imgpoints_Francois, dtype=np.float64)
            self.imgpoints_JS = np.array(self.imgpoints_JS, dtype=np.float64)
            print("shape of imgpoints_Francois is ", tmp.shape, 'len( ) is ', len(tmp))#, 'tpye is ', self.imgpoints_Francois.type())

            self.objpoints = np.array(self.objpoints, dtype=np.float64)
            print("shape of objpoints is ", self.objpoints.shape)#, 'type is ', self.objpoints.type())
            #self.objpoints = np.array([self.opts_loc for i in range(len(self.imgpoints_Francois))])


        print('loadVarInDetection Complete')

    def startNormalStereoCalibration(self):
        global mybalance

        # dataCheck
        tmp = np.array(self.objpoints)
        print('objpoints is', tmp.shape, len(self.objpoints))
        tmp = np.array(self.imgpoints_JS)
        print('imgpoints_JS is ', tmp.shape, len(self.imgpoints_JS))
        tmp = np.array(self.imgpoints_Francois)
        print('self.imgpoints_Francois is ', tmp.shape, len(self.imgpoints_Francois))

        print('width and height is ', (self.width_train_JS, self.height_train_JS))
        # originally works
        N_OK = len(self.objpoints)
        print('N_Ok is ', N_OK)

        self.camera_matrix_JS = np.zeros((3, 3))
        self.camera_matrix_Francois = np.zeros((3, 3))

        self.distCoeffs_JS = np.zeros((4, 1))
        self.distCoeffs_Francois = np.zeros((4, 1))

        self.rotationMatrix = np.zeros((3, 3))  # [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]#
        self.translationVector = np.zeros((3, 1))  # [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)] #

        # self.rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        # self.tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        # retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F
        ret, _, _, _, _, self.rotationMatrix, self.translationVector = cv2.stereoCalibrate(
            objectPoints=self.objpoints, imagePoints1=self.imgpoints_JS, imagePoints2=self.imgpoints_Francois,
            cameraMatrix1=self.camera_matrix_JS, distCoeffs1=self.distCoeffs_JS,
            cameraMatrix2=self.camera_matrix_Francois, distCoeffs2=self.distCoeffs_Francois,
            imageSize=(self.width_train_JS, self.height_train_JS),
            R=None, T=None,  # self.rotationMatrix, T=self.translationVector,
            flags=(cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-3))

        # check the result of calibration
        print('self.camera_matrix_JS is ')
        print(self.camera_matrix_JS)
        print('self.camera_matrix_Francois ')
        print(self.camera_matrix_Francois)

        # print('self.rvecs is ')
        # print(self.rvecs)
        # print('self.tvecs is ')
        # print(self.tvecs)

        print('self.distCoeffs_JS is ')
        print(self.distCoeffs_JS)
        print('self.distCoeffs_Francois is ')
        print(self.distCoeffs_Francois)

        print('RMS re-projection error is ', ret)
        print('balance is ', mybalance)

        print("calibration complete")

    def startSteroCalibration(self):
        global mybalance
        self.imgpoints_Francois = np.array(self.imgpoints_Francois, dtype=np.float64)
        self.imgpoints_JS = np.array(self.imgpoints_JS, dtype=np.float64)
        self.objpoints = np.array(self.objpoints, dtype=np.float64)

        # dataCheck
        tmp = np.array(self.objpoints)
        print('objpoints is', tmp.shape, len(self.objpoints))
        tmp = np.array(self.imgpoints_JS)
        print('imgpoints_JS is ', tmp.shape, len(self.imgpoints_JS))
        tmp = np.array(self.imgpoints_Francois)
        print('self.imgpoints_Francois is ', tmp.shape, len(self.imgpoints_Francois))

        print('width and height is ', (self.width_train_JS, self.height_train_JS))
        # originally works
        N_OK = len(self.objpoints)
        print('N_Ok is ', N_OK)

        self.camera_matrix_JS = np.zeros((3, 3), dtype=np.float64)
        self.camera_matrix_Francois = np.zeros((3, 3), dtype=np.float64)

        self.distCoeffs_JS = np.zeros((4, 1), dtype=np.float64)
        self.distCoeffs_Francois = np.zeros((4, 1), dtype=np.float64)

        self.rotationMatrix = np.zeros((3, 3), dtype=np.float64) # [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]#
        self.translationVector = np.zeros((3, 1), dtype=np.float64) # [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)] #

        #self.rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        #self.tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        ret, _, _, _, _, _, _ = cv2.fisheye.stereoCalibrate(
            objectPoints=self.objpoints, imagePoints1=self.imgpoints_JS, imagePoints2=self.imgpoints_Francois,
            K1=self.camera_matrix_JS, D1=self.distCoeffs_JS,
            K2=self.camera_matrix_Francois, D2=self.distCoeffs_Francois,
            imageSize=(np.array(self.width_train_JS, dtype=np.float64), np.array(self.height_train_JS, dtype=np.float64)),
            R=self.rotationMatrix, T=self.translationVector,
            flags=(cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-3))

        # check the result of calibration
        print('self.camera_matrix_JS is ')
        print(self.camera_matrix_JS)
        print('self.camera_matrix_Francois ')
        print(self.camera_matrix_Francois)

        # print('self.rvecs is ')
        # print(self.rvecs)
        # print('self.tvecs is ')
        # print(self.tvecs)

        print('self.distCoeffs_JS is ')
        print(self.distCoeffs_JS)
        print('self.distCoeffs_Francois is ')
        print(self.distCoeffs_Francois)

        print('RMS re-projection error is ', ret)
        print('balance is ', mybalance)

        print("calibration complete")

    def saveVarAfterCalibration(self):

        with open(nameof_pickel_calibrated_result, 'w') as f:
            pickle.dump([self.camera_matrix_JS, self.distCoeffs_JS, self.camera_matrix_Francois, self.distCoeffs_Francois, self.width_train_JS, self.height_train_JS, self.width_train_Francois, self.height_train_Francois, self.rotationMatrix, self.translationVector], f)

        print('saveVarAfterCalibration complete')

    def loadVarAfterCalibration(self):

        with open(nameof_pickel_calibrated_result) as f:
            self.camera_matrix_JS, self.distCoeffs_JS, self.camera_matrix_Francois, self.distCoeffs_Francois, self.width_train_JS, self.height_train_JS, self.width_train_Francois, self.height_train_Francois, self.rotationMatrix, self.translationVector = pickle.load(f)
            #for old data, erase self.roi, self.map1, self.map2

        print('loadVarAfterCalibration complete')

    def stereoUndistort(self, frame_JS, frame_Francois):

        if self.flag_get_undistort_param == 1:

            frameDim = frame_JS.shape[:2][::-1]
            displayDim = (480, 320)

            self.map1_Francois, self.map2_Francois, self.map1_JS, self.map2_JS = [0, 0, 0, 0]
            self.rotationMatrix_JS, self.rotationMatrix_Francois = [np.eye(3), np.eye(3)]
            self.projectionMatrix_JS, self.projectionMatrix_Francois = [np.eye(3,4), np.eye(3,4)]

            cv2.fisheye.stereoRectify(K1=self.camera_matrix_JS, D1=self.distCoeffs_JS, K2=self.camera_matrix_Francois, D2=self.distCoeffs_Francois,
                                      imageSize=(self.width_train_JS, self.height_train_JS),
                                      R=self.rotationMatrix, tvec=self.translationVector,
                                      flags=cv2.CALIB_ZERO_DISPARITY, R1=self.rotationMatrix_JS, R2=self.rotationMatrix_Francois, P1=self.projectionMatrix_JS, P2=self.projectionMatrix_Francois,
                                      )#Q=None, newImageSize=frameDim, balance=None, fov_scale=1.0)

            self.map1_JS, self.map2_JS = cv2.initUndistortRectifyMap(cameraMatrix=self.camera_matrix_JS, distCoeffs=self.distCoeffs_JS,
                                        R=self.rotationMatrix_JS, newCameraMatrix=self.camera_matrix_JS,
                                        size=displayDim, m1type=cv2.CV_16SC2) #map1=self.map1_JS, map2=self.map2_JS

            self.map1_Francois, self.map2_Francois = cv2.initUndistortRectifyMap(cameraMatrix=self.camera_matrix_Francois, distCoeffs=self.distCoeffs_Francois,
                                        R=self.rotationMatrix_Francois, newCameraMatrix=self.camera_matrix_Francois,
                                        size=displayDim, m1type=cv2.CV_16SC2) # , map1=self.map1_Francois, map2=self.map2_Francois

        frameUndistorted_JS = cv2.remap(src=frame_JS, map1=self.map1_JS, map2=self.map2_JS, interpolation=cv2.INTER_LINEAR, dst=None, borderMode=cv2.BORDER_CONSTANT)
        frameUndistorted_Francois = cv2.remap(src=frame_Francois, map1=self.map1_Francois, map2=self.map2_Francois, interpolation=cv2.INTER_LINEAR, dst=None, borderMode=cv2.BORDER_CONSTANT)
        # cv2.mergeRectification()

        tmp1 = cv2.resize(frameUndistorted_JS, displayDim, cv2.INTER_LINEAR)
        tmp2 = cv2.resize(frameUndistorted_Francois, displayDim, cv2.INTER_LINEAR)
        tmp3 = np.concatenate((tmp1, tmp2), axis=1)

        global flag_show_image
        if flag_show_image == 1:
            cv2.namedWindow('stereoView')
            cv2.imshow('stereoView', tmp3)
            cv2.waitKey(1)

        print('Homography is ')
        print(self.translationVector)
        return tmp1, tmp2, self.translationVector

    #https: // raw.githubusercontent.com / Algomorph / cvcalib / ea157c557f3a8fd02fa01a64023cd017a2851c68 / calib / utils.py
    def compute_stereo_rectification_maps(self, im_size, size_factor):
        new_size = (int(im_size[1] * size_factor), int(im_size[0] * size_factor))
        rotation1, rotation2, pose1, pose2 = \
            cv2.stereoRectify(cameraMatrix1=self.cameras[0].intrinsics.intrinsic_mat,
                              distCoeffs1=self.cameras[0].intrinsics.distortion_coeffs,
                              cameraMatrix2=self.cameras[1].intrinsics.intrinsic_mat,
                              distCoeffs2=self.cameras[1].intrinsics.distortion_coeffs,
                              imageSize=(im_size[1], im_size[0]),
                              R=self.cameras[1].extrinsics.rotation,
                              T=self.cameras[1].extrinsics.translation,
                              flags=cv2.CALIB_ZERO_DISPARITY,
                              newImageSize=new_size
                              )[0:4]
        map1x, map1y = cv2.initUndistortRectifyMap(self.cameras[0].intrinsics.intrinsic_mat,
                                                   self.cameras[0].intrinsics.distortion_coeffs,
                                                   rotation1, pose1, new_size, cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(self.cameras[1].intrinsics.intrinsic_mat,
                                                   self.cameras[1].intrinsics.distortion_coeffs,
                                                   rotation2, pose2, new_size, cv2.CV_32FC1)
        return map1x, map1y, map2x, map2y

    #https://raw.githubusercontent.com/Algomorph/cvcalib/ea157c557f3a8fd02fa01a64023cd017a2851c68/calib/utils.py
    def undistort_stereo(self, test_im_left, test_im_right, size_factor):
        im_size = test_im_left.shape
        map1x, map1y, map2x, map2y = compute_stereo_rectification_maps(self, im_size, size_factor)
        rect_left = cv2.remap(test_im_left, map1x, map1y, cv2.INTER_LINEAR)
        rect_right = cv2.remap(test_im_right, map2x, map2y, cv2.INTER_LINEAR)
        return rect_left, rect_right

    def undistort_imShow(self, frame):
        #print('shape of frames is ', frame.shape)
        #frame = cv2.resize(frame, (964, 1288), cv2.INTER_LINEAR)
        dim0 = frame.shape[:2][::-1]
        dim1 = dim0
        dim2 = dim1
        dim3 = tuple((np.array(dim1)*1).astype(int))
        displayDim = (640, 480) #tuple((np.array(frame.shape[:2][::-1]) / 2.5).astype(int))
        #print('dim3 is ', dim3, 'dim2 is ', dim2, 'dim1 is ', dim1)
        #print('shape of the tained images are ', (self.width, self.height))

        if flag_fisheye_calibrate == 0:

            # best
            # cv2.undistort : https://docs.opencv.org/3.1.0/da/d54/group__imgproc__transform.html#ga69f2545a8b62a6b0fc2ee060dc30559d
            # The function is simply a combination of initUndistortRectifyMap() (with unity R ) and remap() (with bilinear interpolation)
            # I cannot tell the difference between the two line below
            frame_with_new_camera_matrix = cv2.undistort(frame, self.new_camera_matrix, self.distCoeffs, None, None)
            frame_with_origin_camera_matrix = cv2.undistort(frame, self.camera_matrix, self.distCoeffs, None, None)

            # cv2.namedWindow('undistorted frame')
            #cv2.imshow('undistorted frame', frame_with_new_camera_matrix)



            # shit
            frame_with_remap = cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            #cv2.imshow('undistorted frame', frame_with_remap)

            # compare with distorted / undistorted
            # cv2.imshow('undistorted frame', np.concatenate((frame_with_new_camera_matrix, frame), axis=1))
            # compare with camera_matrixes
            # cv2.imshow('undistorted frame', np.concatenate((frame_with_new_camera_matrix, frame_with_origin_camera_matrix), axis=1))
            # test
            cv2.imshow('undistorted frame', np.concatenate((cv2.resize(frame_with_new_camera_matrix, (self.width, self.height), cv2.INTER_CUBIC),
                                                            cv2.resize(frame_with_origin_camera_matrix, (self.width, self.height), cv2.INTER_CUBIC),
                                                            cv2.resize(frame_with_remap, (self.width, self.height), cv2.INTER_CUBIC)
                                                            ),
                                                           axis=1))
            cv2.waitKey(1)

        elif flag_fisheye_calibrate == 1:

            if self.flag_get_undistort_param == 1:

                # self.new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K=self.camera_matrix,
                #                                                                                 D=self.distCoeffs,
                #                                                                                 image_size=(self.width, self.height),
                #                                                                                 R=np.eye(3),
                #                                                                                 P=None,
                #                                                                                 balance=mybalance,
                #                                                                                 new_size=dim3#,fov_scale=1.0
                #                                                                                 )
                #
                # # https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-part-2-13990f1b157f
                # self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(K=self.camera_matrix,
                #                                                              D=self.distCoeffs,
                #                                                              R=np.eye(3),
                #                                                              P=self.camera_matrix,
                #                                                              size=dim3,
                #                                                              m1type=cv2.CV_32FC1)
                #
                #
                # # check the result of calibration
                # print('camera matrix is ')
                # print(self.camera_matrix)
                # print('new camera Matrix is ')
                # print(self.new_camera_matrix)
                # print('distort Coeffs is ')
                # print(self.distCoeffs)
                #
                self.flag_get_undistort_param = 0


            frame_undistorted_fisheye_camera_matrix = cv2.fisheye.undistortImage(frame, self.camera_matrix, self.distCoeffs, Knew=self.camera_matrix, new_size=dim0)#(self.width, self.height))
            # frame_undistorted_fisheye_remap_with_origin_camera_matrix = cv2.remap(src=frame,map1=self.map1,map2=self.map2,interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_DEFAULT)

            if flag_show_image == 1:
                # cv2.imshow('jaesungLensFrame', np.concatenate((cv2.resize(frame_undistorted_fisheye_camera_matrix, displayDim, cv2.INTER_LINEAR),
                #                                          cv2.resize(frame_undistorted_fisheye_remap_with_origin_camera_matrix, displayDim, cv2.INTER_LINEAR),
                #                                          cv2.resize(frame, displayDim, cv2.INTER_LINEAR)),axis=1))

                tmp = cv2.resize(frame_undistorted_fisheye_camera_matrix, displayDim, cv2.INTER_LINEAR)
                cv2.imshow('JS calibrated resuls', tmp)

            cv2.waitKey(1)
            return tmp
        else:
            print('error for <flag_fisheye_calibrate>')
            return None


class dataLoadType:

    flag_fisrt_didLoadVarDetection = 1
    flag_first_didLoadVarCalibration = 1
    flag_first_didHomography = 1

    def __init__(self, singleImage_inst, calibrate_inst):
        self.singleImage_inst = singleImage_inst
        self.calibrate_inst = calibrate_inst
        self.bridge = CvBridge()

    def __init__(self, singleImage_JS, singleImage_Francois, calibrate_inst, singleImage_cctv = None):
        self.singleImage_JS = singleImage_JS
        self.singleImage_Francois = singleImage_Francois
        self.calibrate_inst = calibrate_inst
        self.bridge = CvBridge()

        global flag_cctv_run
        if flag_cctv_run == 1:
            self.singleImage_cctv = singleImage_cctv
        else:
            self.singleImage_cctv = None

    def subscribeImage(self):
        print('start to subscribe image')
        #rospy.init_node('dataLoadType', anonymous=True)
        self.rospySubImg = rospy.Subscriber(ROS_TOPIC, Image, self.callback)
        #automatically go to the callback function : self.callback()

    def subscribeMultiImage_Sync(self):
        print('start to subscribe multi iamges')
        subImg_JS = message_filters.Subscriber('calibration_JS_fisheye/image_calibrated', Image)
        subImg_Francois = message_filters.Subscriber('calibration_Francois_fisheye/image_calibrated', Image)
        # subImg_JS = message_filters.Subscriber('jaesung_lens_camera/image_color', Image)
        # subImg_Francois = message_filters.Subscriber('francois_lens_camera/image_color', Image)
        tss = message_filters.TimeSynchronizer([subImg_JS, subImg_Francois], 1000)
        tss.registerCallback(self.callback)

    def subscribeMultiImage_Async(self):
        print('start to subscribe Multi images asynchronously')



        global flag_cctv_run
        if flag_cctv_run == 1:

            subImg_JS = message_filters.Subscriber('calibration_JS_fisheye/image_calibrated', Image)
            subImg_Francois = message_filters.Subscriber('calibration_Francois_fisheye/image_calibrated', Image)
            subImg_cctv = message_filters.Subscriber('cctv_camera/image_republished', Image)
            ts = message_filters.ApproximateTimeSynchronizer([subImg_JS, subImg_Francois, subImg_cctv], 10, 5)

            ts.registerCallback(self.callback)
            print('fuck')
        else:
            subImg_JS = message_filters.Subscriber('calibration_JS_fisheye/image_calibrated', Image)
            subImg_Francois = message_filters.Subscriber('calibration_Francois_fisheye/image_calibrated', Image)
            # subImg_JS = message_filters.Subscriber('jaesung_lens_camera/image_color', Image)
            # subImg_Francois = message_filters.Subscriber('francois_lens_camera/image_color', Image)
            ts = message_filters.ApproximateTimeSynchronizer([subImg_JS, subImg_Francois], 10 , 5)
            ts.registerCallback(self.callback)

    def stopSubscribing(self):
        print('stop subscribing image')
        #self.image_sub.shutdown()
        self.rospySubImg.unregister()

    def loadImageInFiles(self):
        global fileList
        fileList = glob.glob(path_image_database)
        fileList = sorted(fileList)
        print('path_image_database is ', path_image_database)

        global num_of_image_in_database
        num_of_image_in_database = len(fileList)
        print('what is fileList', fileList, 'num is ', num_of_image_in_database)

        global count
        count = 0
        for i in range(0, num_of_image_in_database, 2):
            count = count + 2
            print('The ', i, '-th image is under processing', 'count is ', count, 'and ', fileList[i])
            self.singleImage_JS.saveImage(cv2.imread(fileList[i]))
            self.singleImage_Francois.saveImage(cv2.imread(fileList[i+1]))
            self.wrapper()


    def publishImage(self):
        # http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscriber%28python%29
        print('start to publish JS image')
        self.rospyPubImg = rospy.Publisher('homography/image_calibrated', Image, queue_size=10)
        #rospy.init_node('calibrated_JS_lens_fisheye', anonymous=True)
        rate = rospy.Rate(10)  # 10Hz

    def callback(self, data_JS=None, data_Francois=None, data_cctv=None):
        global count
        try:
            print('>> into callback function')
            # parse message into image
            # bgr8: CV_8UC3, color image with blue-green-red color order and 8b

            self.singleImage_JS.saveImage(self.bridge.imgmsg_to_cv2(data_JS, "bgr8"))
            self.singleImage_Francois.saveImage(self.bridge.imgmsg_to_cv2(data_Francois, "bgr8"))

            global flag_cctv_run
            if flag_cctv_run == 1:
                img_cctv = self.bridge.imgmsg_to_cv2(data_cctv, "bgr8")
                self.singleImage_cctv.saveImage(img_cctv)
                # cv2.imshow('check the cctv display', img_cctv)
                # cv2.waitKey(1)

            # if you want to work asynchronously, edit the lines below
            count = count + 2



            # for i in range(250):
            #     for j in range(250):
            #         for k in range(3):
            #             if self.singleImage_Francois.imgData[i, j, k] == 0:
                            # if self.singleImage_JS.imgData[i, j, k] == 0:
                            #     self.singleImage_Francois.imgData[i, j, k] = self.singleImage_cctv.imgData[i, j, k]
                            # else:
                            #self.singleImage_Francois.imgData[i, j, k] = self.singleImage_JS.imgData[i, j, k]


                                #self.singleImage_JS.imgData[i, j, k] = self.singleImage_cctv.imgData[i, j, k]

            # cv2.namedWindow('simulation')
            # cv2.imshow('simulation', cv2.resize(self.singleImage_JS.imgData, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR))



            # alpha blending
            # alpha = 0.5
            # beta = (1 - alpha)
            # self.singleImage_Francois.imgData = cv2.addWeighted(src1=self.singleImage_Francois.imgData, alpha=alpha, src2=self.singleImage_cctv.imgData, beta=beta, gamma=0.0, dst=None, dtype=-1)
            # cv2.namedWindow(str('alphaBlended alpha = ' + str(alpha) + ' beta = ' + str(beta)))
            # cv2.imshow(str('alphaBlended alpha = ' + str(alpha) + ' beta = ' + str(beta)), cv2.resize(alphaBlended, (0,0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR))
            #
            # cv2.waitKey(1)

            #self.wrapper_homography(self.singleImage_JS.imgData, self.singleImage_Francois.imgData, frame_cctv=self.singleImage_cctv.imgData)




            # self.wrapper()

            # cv2.imshow('multi image display', np.concatenate((cv2.resize(self.singleImage_JS.imgData, (480, 320), cv2.INTER_LINEAR),
            #                                                   cv2.resize(self.singleImage_Francois.imgData, (480, 320), cv2.INTER_LINEAR)), axis=1))
            # cv2.waitKey(1)

            # cv2.imshow('fuck', self.singleImage_JS.imgData[:303])
            # cv2.namedWindow('stitched image')
            # cv2.imshow('stitched image', np.concatenate((self.singleImage_Francois.imgData[:, :382], self.singleImage_JS.imgData[:, 339:]), axis=1))
            # cv2.waitKey(1)
            #
            # if flag_save_image_onlyWhichDetectCheckeboard == 1: #flag_subscribe_new_image_not_load_old_image == 1 and
            #     print('saved')
            #     cv2.imwrite(save_video_to_the_path + (str((count + 10000)) + '.png'), self.singleImage_JS.imgData)
            #     count = count + 1
            #     cv2.imwrite(save_video_to_the_path + (str((count + 10000)) + '.png'), self.singleImage_Francois.imgData)
            #     count = count + 1

            if flag_publish_calibrated_image == 1:
                try:
                    self.rospyPubImg.publish(self.bridge.cv2_to_imgmsg(self.singleImage_Francois.imgData, "bgr8"))
                    cv2.waitKey(1)
                except CvBridgeError as e:
                    print(e)
                    cv2.waitKey(1)



        except CvBridgeError as e:
            print(e)

    # def callback(self, data):
    #     #print('come to callback function')
    #     global count
    #     try:
    #         # parse message into image
    #         # bgr8: CV_8UC3, color image with blue-green-red color order and 8bit
    #         self.singleImage_inst.saveImage(self.bridge.imgmsg_to_cv2(data, "bgr8"))
    #         count = count + 1
    #flag_first_didHomography
    #         # if you want to work asynchronously, edit the lines below
    #         self.wrapper()
    #
    #     except CvBridgeError as e:
    #         print(e)]

    def wrapper_homography(self, frame_JS=None, frame_Francois=None, frame_cctv=None):

        if self.flag_first_didHomography == 1:

            if frame_Francois != None:
                # Francois
                srcPoints_Francois = np.array([[78, 63], [213, 54], [370, 53], [547, 45],
                                               [546, 80],
                                               [547, 112],
                                               [64, 143], [199, 143], [356, 143], [536, 145],

                                               [49, 225],

                                               [35, 314], [166, 321], [331, 339], [518, 356],
                                               [506, 394],
                                               [508, 427],
                                               [13, 401], [154, 417], [314, 438], [502, 465]])

                dstPoints_Francois = np.array([[0, 0], [0, 50], [0, 100], [0, 150],
                                               [16, 150],
                                               [34, 150],
                                               [50, 0], [50, 50], [50, 100], [50, 150],

                                               [100, 0],

                                               [150, 0], [150, 50], [150, 100], [150, 150],
                                               [166, 150],
                                               [184, 150],
                                               [200, 0], [200, 50], [200, 100], [200, 150]])

                srcPoints_Francois, dstPoints_Francois = np.array(srcPoints_Francois), np.array(dstPoints_Francois)
                print('dstPoints_Francois is ', dstPoints_Francois.shape, srcPoints_Francois.shape)

                self.homography_Francois, mask = cv2.findHomography(srcPoints=srcPoints_Francois, dstPoints=dstPoints_Francois, method=cv2.RANSAC)
                print('homography_Francois is ', self.homography_Francois)
                # self.homography_Francois = cv2.getPerspectiveTransform(srcPoints_Francois, dstPoints_Francois)

            if frame_JS is not None:
                # Jaesung
                # Jaesung
                srcPoints_JS = np.array([[68, 133], [149, 129], [253, 118], [372, 107], [515, 100],
                                         [376, 126],
                                         [372, 153],
                                         [133, 185], [239, 183], [378, 181], [524, 173],
                                         [125, 241], [233, 244], [367, 250], [531, 256],
                                         [20, 309], [106, 321], [227, 333], [369, 343], [539, 359],
                                         [365, 371],
                                         [363, 395],
                                         [13, 376], [103, 399], [222, 214], [365, 439], [544, 460]])
                dstPoints_JS = np.array([[0, 0], [0, 50], [0, 100], [0, 150], [0, 200],
                                         [16, 150],
                                         [33, 150],
                                         [50, 50], [50, 100], [50, 150], [50, 200],
                                         [100, 50], [100, 100], [100, 150], [100, 200],
                                         [150, 0], [150, 50], [150, 100], [150, 150], [150, 200],
                                         [166, 150],
                                         [184, 150],
                                         [200, 0], [200, 50], [200, 100], [200, 150], [200, 200]])

                srcPoints_JS, dstPoints_JS = np.array(srcPoints_JS), np.array(dstPoints_JS)
                self.homography_JS, mask = cv2.findHomography(srcPoints=srcPoints_JS, dstPoints=dstPoints_JS, method=cv2.RANSAC)

            global flag_cctv_run
            if flag_cctv_run == 1 and frame_cctv is not None:
                srcPoints_cctv = np.array([[102, 234], [182, 221], [262, 218],
                                           [92, 266], [185, 258], [280, 249],
                                           [78, 313], [188, 303], [300, 295],
                                           [60, 382], [197, 374], [241, 371], [285, 366], [334, 360],
                                           [36, 492], [210, 481], [377, 461], [515, 432],
                                           [16, 648], [231, 643], [440, 600]])

                dstPoints_cctv = np.array([[250, 0], [200, 0], [150, 0],
                                           [250, 50], [200, 50], [150, 50],
                                           [250, 100], [200, 100], [150, 100],
                                           [250, 150], [200, 150], [184, 150], [166, 150], [150, 150],
                                           [250, 200], [200, 200], [150, 200], [100, 200],
                                           [250, 250], [200, 250], [150, 250]])

                srcPoints_cctv, dstPoints_cctv = np.array(srcPoints_cctv), np.array(dstPoints_cctv)
                self.homography_cctv, mask = cv2.findHomography(srcPoints=srcPoints_cctv, dstPoints=dstPoints_cctv, method=cv2.RANSAC)

            self.flag_first_didHomography = 0

        if frame_Francois is not None:
            frame_homography_Francois = cv2.warpPerspective(frame_Francois, self.homography_Francois, (250, 250))

        if frame_JS is not None:
            frame_homography_JS = cv2.warpPerspective(frame_JS, self.homography_JS, (250, 250))

        if frame_cctv is not None:
            frame_homography_cctv = cv2.warpPerspective(frame_cctv, self.homography_cctv, (250, 250))

        # cv2.namedWindow('frame_homography_Francois results')
        # cv2.imshow('frame_homography_Francois results', cv2.resize(frame_homography_Francois, (0,0), fx=2, fy=2))
        # cv2.waitKey(1)
        #
        # cv2.namedWindow('frame_homography_JS results')
        # cv2.imshow('frame_homography_JS results', cv2.resize(frame_homography_JS, (0,0), fx=2, fy=2))
        # cv2.waitKey(1)

        for i in range(250):
            for j in range(250):
                for k in range(3):
                    if frame_homography_JS[i, j, k] < 10:
                        frame_homography_JS[i, j, k] = frame_homography_cctv[i, j, k]




        #alpha blending
        alpha = 0
        beta = (1-alpha)
        #alphaBlended = cv2.addWeighted(src1=frame_homography_cctv, alpha=alpha, src2=frame_homography_JS, beta=beta, gamma=0.0, dst=None, dtype=-1)
        #cv2.namedWindow(str('alphaBlended alpha = ' + str(alpha) + ' beta = ' + str(beta)))
        #cv2.imshow(str('alphaBlended alpha = ' + str(alpha) + ' beta = ' + str(beta)), cv2.resize(alphaBlended, (0,0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR))
        cv2.imshow('simulation', cv2.resize(frame_homography_JS, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR))



        # tmp = cv2.resize(cv2.flip(frame_homography_JS, cv2.ROTATE_180), (0, 0), fx=2, fy=2)
        # cv2.imshow('concatenated', tmp)

        # cv2.namedWindow('concatenated')
        # cv2.imshow('concatenated', np.concatenate((cv2.resize(frame_homography_Francois, (0,0), fx=2, fy=2), cv2.flip(tmp[300:, :, :], cv2.ROTATE_180)), axis = 0))

        cv2.waitKey(1)

    def wrapper(self):
        global count

        if flag_load_detected_result == 0 and flag_load_calibrated_result == 0:
            print('count is ', count, 'and', 'num_of_image_in_database is ', num_of_image_in_database,'in wrapper')
            if count < num_of_image_in_database:
                self.calibrate_inst.detectSteroCheckerBoard(self.singleImage_JS.imgData, self.singleImage_Francois.imgData)
            elif count == num_of_image_in_database:
                print('?')
                self.calibrate_inst.detectSteroCheckerBoard(self.singleImage_JS.imgData, self.singleImage_Francois.imgData)
                self.calibrate_inst.saveVarInDetection()
                self.calibrate_inst.startSteroCalibration()
                self.calibrate_inst.saveVarAfterCalibration()
            else:
                print('??')
                self.singleImage_JS.imgCalibrated, self.singleImage_Francois.imgCalibrated, Homography = self.calibrate_inst.stereoUndistort(self.singleImage_JS.imgData, self.singleImage_Francois.imgData)

        elif flag_load_detected_result == 1 and flag_load_calibrated_result == 0:
            # load Detection results, start calibratin and show undistorted image
            if self.flag_fisrt_didLoadVarDetection == 1:
                self.calibrate_inst.loadVarInDetection()
                self.calibrate_inst.startSteroCalibration()
                #self.calibrate_inst.startNormalStereoCalibration()
                self.calibrate_inst.saveVarAfterCalibration()
                # do not come here again
                self.flag_fisrt_didLoadVarDetection = 0

                self.singleImage_JS.imgCalibrated, self.singleImage_Francois.imgCalibrated, Homography = self.calibrate_inst.stereoUndistort(self.singleImage_JS.imgData, self.singleImage_Francois.imgData)

        elif flag_load_calibrated_result == 1:
            # load Calibration results and show undistorted image
            if self.flag_first_didLoadVarCalibration == 1:
                self.calibrate_inst.loadVarAfterCalibration()
                # do not come here again
                self.flag_first_didLoadVarCalibration = 0

            self.singleImage_JS.imgCalibrated, self.singleImage_Francois.imgCalibrated, Homography = self.calibrate_inst.stereoUndistort(self.singleImage_JS.imgData, self.singleImage_Francois.imgData)

        else:
            print("fucking error for count = ", count)


        s = Stitch(self.singleImage_JS.imgData, self.singleImage_Francois.imgData)
        # s = Stitch(self.singleImage_JS.imgCalibrated, self.singleImage_Francois.imgCalibrated)
        s.leftshift(Homography)
        # s.showImage('left')

        s.rightshift(Homography)
        print('done')

        cv2.namedWindow('stitch result')
        cv2.imshow('stitch result', s.leftImage)
        cv2.waitKey(1)

class singleImageData:
    height = 0
    width = 0
    imgData = None
    imgCalibrated = None
    imgDetected = None

    def saveImage(self, img):
            self.imgData = img
                #cv2.resize(img, (320, 480), cv2.INTER_LINEAR)#(482, 644)
            #self.imgDetected = img.copy()
            self.height, self.width = img.shape[::2]#
            # (320, 480)


    def resize(self, ratio):
        #cv2.resize(self.imgData, )
        print('resize of the image completed')

if __name__ == '__main__':

    print("check the version opencv.")
    print(cv2.__version__)

    singleImage_JS = singleImageData()
    singleImage_Francois = singleImageData()
    calibrate_inst = calibration()

    global flag_cctv_run
    if flag_cctv_run == 1:
        singleImage_cctv = singleImageData()
    else:
        singleImage_cctv = None

    dataLoadType_inst = dataLoadType(singleImage_JS, singleImage_Francois, calibrate_inst, singleImage_cctv=singleImage_cctv)

    rospy.init_node('imageStitching', anonymous=True)
    try:
        if flag_subscribe_new_image_not_load_old_image == 1:
            dataLoadType_inst.subscribeMultiImage_Async()
            #dataLoadType_inst.subscribeMultiImage_Sync()

            if flag_publish_calibrated_image == 1:
                dataLoadType_inst.publishImage()

            rospy.spin()
        else:
            dataLoadType_inst.loadImageInFiles()
    except KeyboardInterrupt:
        print('end')


