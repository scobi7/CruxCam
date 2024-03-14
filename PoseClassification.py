import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import math

# Initialize MediaPipe Pose class and the pose functions
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils

def detectPose(frame, pose, display=True):
    output_frame = frame.copy()  # Create a copy of the frame
    # Convert frame to RGB color space
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process the frame with MediaPipe Pose
    results = pose.process(frame_rgb)
    # Get frame dimensions
    height, width, _ = frame.shape
    landmarks = []

    # Draw pose landmarks on the output frame if detected
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image=output_frame, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)

        # Convert landmarks to image coordinates
        for landmark in results.pose_landmarks.landmark:
            landmarks.append((int(landmark.x * width), int(landmark.y * height), int(landmark.z * width)))

    # Display the output frame if display flag is set to True
    if display:
        cv2.imshow("Pose Detection", output_frame)
        cv2.waitKey(1)

    # Return the output frame and detected landmarks
    if results.pose_landmarks:
        return output_frame, landmarks
    else:
        return None, None


def calculateAngle(landmark1, landmark2, landmark3):
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    if angle < 0:
        angle += 360
    return angle


def PoseClassification(landmarks, output_image, display=False):
    label = 'climbing'
    color = (0,0,225)

    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    
    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])

    if left_elbow_angle < 90:
        label = 'left locked off'
        if left_elbow_angle < 90 and right_elbow_angle < 90:
            label = 'both arms locked off'
    
    if right_elbow_angle < 90:
        label = 'right locked off'
        if right_elbow_angle < 90 and right_elbow_angle < 90:
            label = 'both arms locked off'

    if label != 'climbing':
        color = (0,225,0)
    # Draw the label on the output image
    cv2.putText(output_image, label, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

    if display:
        cv2.imshow("Pose Classification", output_image)
        cv2.waitKey(1)
    else:
        return output_image, label

def main():
    # Initialize video capture
    pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
    video_path = "posevids/BrandonRDK.AVI"  # Update with your video file path
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        success, frame = cap.read()  # Read a frame
        if not success:
            break
        
        # Process the frame
        frame, landmarks = detectPose(frame, pose_video, display=False)
        if landmarks:
            frame, _ = PoseClassification(landmarks, frame, display=False)

        # Display the frame
        cv2.imshow("Video", frame)

        # Set the delay to achieve approximately 0.6x speed
        delay = int(1000 / 24 * 0.6)  # Assuming 24 frames per second
        # Adjust the above formula according to your video's frame rate

        # Check for user input to quit
        if cv2.waitKey(delay) & 0xFF == 27:  # Press 'Esc' to exit
            break
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    pose.close()


if __name__ == "__main__":
    main()
