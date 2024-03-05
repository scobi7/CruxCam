import cv2 as cv
import mediapipe as  mp 
import time
import PoseModule as pm

def main():
    cap = cv.VideoCapture('posevids/DevinDropKnee.MOV')
    pTime = 0
    detector = pm.poseDetector()
    while True:
        #loading video with frame rate information and color
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)


        #from here to the cv2.circle, I am iterating to only print/ track a specific point on the body. For example, 14 is the right elbow so it will  be 
        #highlited in the video and those values/coordinates will be printed. Consider which values are most important for the climbing
        if len(lmList) !=0:
            print(lmList[14])
            cv2.circle(img, (lmList[14][1], lmList[14][2]) 15, (0, 0, 255), cv2.FILLED)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv.putText(img, str(int(fps)), (70,50), cv.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
        cv.imshow('Image', img)
        cv.waitKey(1)
