import cv2 as cv
import mediapipe as  mp 
import time

#creating poses
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv.VideoCapture('posevids/DevinDropKnee.MOV')
pTime = 0
while True:
    #loading video with frame rate information and color
    success, frames = cap.read()
    framesRGB = cv.cvtColor(frames, cv.COLOR_BGR2RGB)
    results = pose.process(framesRGB)

    #this will generate numbers for x,y and z axis' as well as visibility
    #could I use this to generate the angles of the limbs?
    #print (results.pose_landmarks)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(frames, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            print(id, lm)
            cx, cy = int(lm.x*w), int(lm.y*h)
            cv2.circle(img, (cx, cy), 4, (255,10,10) cv2.FILLED)
            
    

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(frames, str(int(fps)), (70,50), cv.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
    cv.imshow('Image', frames)
    cv.waitKey(1)