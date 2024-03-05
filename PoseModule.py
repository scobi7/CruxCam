import cv2 as cv
import mediapipe as  mp 
import time

class poseDetector():

    def __init__(self, mode = False, upBody = False, smooth = True,
                 detectionCon = 0,5, trackCon = 0.5):
        
        self.mode = mode
        self.upBody = upBody
        self.smooth =  smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        #creating poses
        self. mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, self.detectionCon self.trackCon)
        

    def findPose(self, img, draw = True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        #this will generate numbers for x,y and z axis' as well as visibility
        #could I use this to generate the angles of the limbs?
        #print (results.pose_landmarks)
        if draw:
            if self.results.pose_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img
    
    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                #print(id, lm)
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 4, (255,10,10) cv2.FILLED)
        return lmList
                

def main():
    cap = cv.VideoCapture('posevids/DevinDropKnee.MOV')
    pTime = 0
    detector = poseDetector()
    while True:
        #loading video with frame rate information and color
        success, img = cap.read()
        img = detector.findPose(img)

        #from here to the cv2.circle, I am iterating to only print/ track a specific point on the body. For example, 14 is the right elbow so it will  be 
        #highlited in the video and those values/coordinates will be printed. Consider which values are most important for the climbing
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

if __name__ == "__main__":
    main()