# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 19:37:24 2021

@author: manus
"""

from cvzone.FaceDetectionModule import FaceDetector
from cvzone.PoseModule import PoseDetector
from cvzone.HandTrackingModule import HandDetector
from cvzone.FaceMeshModule import FaceMeshDetector
import cv2
import os


def squelet_track(ip="http://192.168.1.20:8080/video"):

    cap = cv2.VideoCapture(ip)
    frame_width = int(cap.get(3))

    frame_height = int(cap.get(4))

    out = cv2.VideoWriter('cvz.avi', cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
    detector = PoseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False)
        if bboxInfo:
            center = bboxInfo["center"]
            cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)
            out.write(img)
            cv2.imshow("Image", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def face_track(ip="http://192.168.1.20:8080/video"):
    cap = cv2.VideoCapture(ip)
    detector = FaceMeshDetector(maxFaces=2)
    detector2 = FaceDetector()
    while True:
        success, img = cap.read()
        detector2.findFaces(img)

        detector.findFaceMesh(img)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


face_track(0)
