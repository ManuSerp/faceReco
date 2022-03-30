
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.PoseModule import PoseDetector
from cvzone.HandTrackingModule import HandDetector
from cvzone.FaceMeshModule import FaceMeshDetector
import cv2
from model import *


def saveData(mat):
    f = open("soft_dat/mesh.dat", "w")
    f.write(str(mat))
    f.write("\n")
    f.close()


def face_track(ip="http://192.168.1.20:8080/video"):
    pres = ["manu", "pierre"]
    cap = cv2.VideoCapture(ip)
    detector = FaceMeshDetector(maxFaces=2)
    detector2 = FaceDetector()
    cnn = idmodel()
    cnn.load()
    font = cv2.FONT_HERSHEY_TRIPLEX

    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (1, 255, 1)
    thickness = 1
    lineType = 2

    while True:
        success, img = cap.read()
        detector2.findFaces(img)

        a, b = detector.findFaceMesh(img)

        saveData(b)
        detec = genUsable("soft_dat/mesh.dat")
        prs = cnn.pred(detec)
        var = "None"
        if prs != None:

            var = pres[prs[0][0]]

        cv2.putText(img, var,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


face_track(0)
