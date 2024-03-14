
def getDistanceX(landmark1, landmark2):
    x1 = landmark1
    x2 = landmark2
    
    distance_x = abs(x1 - x2)
    return distance_x

def getDistanceY(landmark1, landmark2):
    y1=landmark1
    y2= landmark2

    distance_y = abs(y1 - y2)
    return distance_y


def PoseClassification(mp_pose, landmarks, output_image, display=False):
        
    label = "climbing"
    color = (0, 225, 0)

    right_dist_x = getDistanceX(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value], landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value])
    right_dist_y = getDistanceY(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value], landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value])

    # Check if both x and y distances are less than or equal to the threshold
    x_threshold = 20
    y_threshold = 20

    if right_dist_x <= x_threshold and right_dist_y <= y_threshold:
        label = "dropknee"
        color = (0, 0, 225)


    # Draw the label on the output image
    cv2.putText(output_image, label, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

    if display:
        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:, :, ::-1]), plt.title("Output Image")
        plt.axis('off')
        plt.show()
    else:
        return output_image, label
        


def main():
    # Initializing MediaPipe Pose class and the pose functions
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.3, model_complexity=2)
    mp_drawing = mp_pose.solutions.drawing_utils

    # Import a video reader
    video_path = "posevids/MalachiRDK.AVI"  # Update with your video file path
    cap = cv2.VideoCapture(video_path)

    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    
    frame_count = 0  # Initialize frame count for saving frames
    output_folder = "VidFrames"  # Output folder for saving frames

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Process the frame
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_height, img_width, _ = frame.shape
        img_copy = frame.copy()

        if results.pose_landmarks:
            # Extract landmarks from index 10 to 28 (inclusive)
            #specific_landmarks = results.pose_landmarks.landmark[10:29]

            mp_drawing.draw_landmarks(image=img_copy, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS)

            for i in range(23, 29):
                landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value]

                x = int(landmark.x * img_width)
                y = int(landmark.y * img_height)
                z = int(landmark.z * img_width)
                visibility = landmark.visibility

                print(f'{mp_pose.PoseLandmark(i).name}:')
                print(f'x:{x}:')
                print(f'y:{y}:')
                print(f'z:{z}:')
                print(f'visibility:{visibility}:')

                # Draw a circle at the landmark position
                cv2.circle(img_copy, (int(x), int(y)), 5, (0, 255, 0), -1)

            # Label pose as climbing
            img_copy, label = PoseClassification(mp_pose, results.pose_landmarks, img_copy)

            # Save the frame as an image to the specified folder
            frame_count += 1
            frame_filename = os.path.join(output_folder, f"FRAME_{frame_count:04d}.png")
            cv2.imwrite(frame_filename, img_copy)
            print(f"SAVED: {frame_filename}")

        # Display the frame and 3D plot
        cv2.imshow("Pose Detection", img_copy)
        plt.pause(0.01)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    pose.close()

if __name__ == "__main__":
    main()


def main():
    # Initialize video capture
    #pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
    video_path = "posevids/MalachiRDK.AVI"  # Update with your video file path
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        success, frame = cap.read()  # Read a frame
        if not success:
            break

        # Process the frame
        #frame, landmarks = detectPose(frame, pose_video, display=False)
        #if landmarks:
            #frame, _ = PoseClassification(landmarks, frame, display=False)

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


