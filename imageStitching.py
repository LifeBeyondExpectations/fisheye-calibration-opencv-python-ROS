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

#paramter
count = 0

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


class Stitch:
    def __init__(self, args):
        self.images = [cv2.resize(cv2.imread(each), (480, 320)) for each in args]
        self.count = len(self.images)
        print('what is self.count ', self.count)
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


    def leftshift(self):
        # self.left_list = reversed(self.left_list)
        a = self.left_list[0]
        for b in self.left_list[1:]:
            H = self.matcher_obj.match(a, b, 'left')
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

    def rightshift(self):
        for each in self.right_list:
            H = self.matcher_obj.match(self.leftImage, each, 'right')
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


class dataLoadType:

    flag_fisrt_didLoadVarDetection = 1
    flag_first_didLoadVarCalibration = 1

    def __init__(self, singleImage_inst, calibrate_inst):
        self.singleImage_inst = singleImage_inst
        self.calibrate_inst = calibrate_inst
        self.bridge = CvBridge()

    def __init__(self, singleImage_JS, singleImage_Francois, calibrate_inst=None):
        self.singleImage_JS = singleImage_JS
        self.singleImage_Francois = singleImage_Francois
        # self.calibrate_inst = calibrate_inst
        self.bridge = CvBridge()

    def subscribeImage(self):
        print('start to subscribe image')
        #rospy.init_node('dataLoadType', anonymous=True)
        self.rospySubImg = rospy.Subscriber(ROS_TOPIC, Image, self.callback)
        #automatically go to the callback function : self.callback()

    def subscribeMultiImage_Sync(self):
        print('start to subscribe multi iamges')
        tss = TimeSynchronizer(message_filters.Subscriber('calibration_JS_fisheye/image_calibrated', Image),
                               message_filters.Subscriber('calibration_Francois_fisheye/image_calibrated', Image))
        tss.registerCallback(self.callback)

    def subscribeMultiImage_Async(self):
        print('start to subscribe Multi images asynchronously')
        subImg_JS = message_filters.Subscriber('calibration_JS_fisheye/image_calibrated', Image)
        subImg_Francois = message_filters.Subscriber('calibration_Francois_fisheye/image_calibrated', Image)
        ts = message_filters.ApproximateTimeSynchronizer([subImg_JS, subImg_Francois], 10 , 5)
        ts.registerCallback(self.callback)

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
        #rospy.init_node('calibrated_JS_lens_fisheye', anonymous=True)
        rate = rospy.Rate(10)  # 10Hz

    def callback(self, data_JS, data_Francois):
        print('>> come to callback function for multi messages')
        global count
        try:
            # parse message into image
            # bgr8: CV_8UC3, color image with blue-green-red color order and 8bit
            self.singleImage_JS.saveImage(self.bridge.imgmsg_to_cv2(data_JS, "bgr8"))
            self.singleImage_Francois.saveImage(self.bridge.imgmsg_to_cv2(data_Francois, "bgr8"))
            count = count + 1

            # if you want to work asynchronously, edit the lines below
            #self.wrapper()
            cv2.imshow('multi image display', np.concatenate((cv2.resize(self.singleImage_JS.imgData, (300,300), cv2.INTER_LINEAR),
                                                                cv2.resize(self.singleImage_Francois.imgData, (300, 300), cv2.INTER_LINEAR))
                                                                , axis=1))
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
    #
    #         # if you want to work asynchronously, edit the lines below
    #         self.wrapper()
    #
    #     except CvBridgeError as e:
    #         print(e)

    def wrapper(self):
        global count, num_of_image_in_database
        global flag_subscribe_new_image_not_load_old_image, flag_load_calibrated_result

        # fucking code
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

    def resize(self, ratio):
        #cv2.resize(self.imgData, )
        print('resize of the image completed')

if __name__ == '__main__':

    print("check the version opencv.")
    print(cv2.__version__)

    singleImage_JS = singleImageData()
    singleImage_Francois = singleImageData()
    #calibrate_inst = calibration()
    dataLoadType_inst = dataLoadType(singleImage_JS, singleImage_Francois, None)

    rospy.init_node('imageStitching', anonymous=True)
    try:
        dataLoadType_inst.subscribeMultiImage_Async()
        rospy.spin()
    except KeyboardInterrupt:
        print('end')


    # fileList = glob.glob('stitchTest/*.png')
    # print('fileList is ', fileList)
    #
    # s = Stitch(fileList)
    # s.leftshift()
    # # s.showImage('left')
    #
    # s.rightshift()
    # print('done')
    #
    # cv2.imshow('stitch result', s.leftImage)
    # cv2.waitKey(1)

    #cv2.imwrite("test12.jpg", s.leftImage)
    #print('image written')

