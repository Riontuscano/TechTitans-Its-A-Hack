import cv2
import time
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cvlib as cv
import mediapipe as mp
import os
from Help_Detector import HandTracking as htm

# Initialize mediapipe for hand detection and tracking
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Class for detecting SOS hand gesture
class HandGestureDetector:
    def __init__(self, wCam=640, hCam=480, detectionCon=0.75):
        self.wCam = wCam
        self.hCam = hCam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Camera failed to initialize.")
            exit()

        self.cap.set(3, self.wCam)
        self.cap.set(4, self.hCam)
        self.detector = htm.handDetector(detectionCon=detectionCon)
        self.tipIds = [4, 8, 12, 16, 20]
        self.pTime = 0
        self.gesture_sequence = []
        self.help_detected = False
        self.help_start_time = 0
        self.help_duration = 3  # seconds
        self.flash_interval = 0.5
        self.last_flash_time = 0
        self.show_help = True

    def is_help_sign(self, lmList):
        """Detect the SOS gesture sequence: Open Hand -> Thumb Curled -> Fist."""
        if len(lmList) == 0:
            return None
        thumb_up = lmList[self.tipIds[0]][1] > lmList[self.tipIds[0] - 1][1]
        fingers_up = [lmList[self.tipIds[id]][2] < lmList[self.tipIds[id] - 2][2] for id in range(1, 5)]
        if all(fingers_up) and thumb_up:
            return "Open Hand"
        elif not thumb_up and all(fingers_up):
            return "Thumb Curled"
        elif not any(fingers_up):
            return "Fist"
        return None

    def check_for_help_sequence(self):
        return self.gesture_sequence == ["Open Hand", "Thumb Curled", "Fist"]

    def run(self, frame):
        img = self.detector.findHands(frame)
        lmList = self.detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            gesture = self.is_help_sign(lmList)
            if gesture:
                if len(self.gesture_sequence) == 0 or gesture != self.gesture_sequence[-1]:
                    self.gesture_sequence.append(gesture)
                if len(self.gesture_sequence) > 3:
                    self.gesture_sequence.pop(0)
                if self.check_for_help_sequence():
                    self.help_detected = True
                    self.help_start_time = time.time()
                    self.gesture_sequence.clear()
            if gesture:
                cv2.putText(img, gesture, (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        if self.help_detected:
            elapsed_time = time.time() - self.help_start_time
            if elapsed_time <= self.help_duration:
                if time.time() - self.last_flash_time >= self.flash_interval:
                    self.show_help = not self.show_help
                    self.last_flash_time = time.time()
                if self.show_help:
                    cv2.putText(img, "HELP", (200, 400), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 5)
            else:
                self.help_detected = False
        return img

# Load the violence detection model
violence_model = load_model('TechTitans-Its-A-Hack/Wivsafe_Detection/Violence_Detector/violence_detection.keras')
classes = ['non-violent', 'violent']

# Load gender detection models
model_base_path_for_gender = "TechTitans-Its-A-Hack/Wivsafe_Detection/Gender_Detector/"
faceProto = os.path.join(model_base_path_for_gender, "opencv_face_detector.pbtxt")
faceModel = os.path.join(model_base_path_for_gender, "opencv_face_detector_uint8.pb")
genderProto = os.path.join(model_base_path_for_gender, "gender_deploy.prototxt")
genderModel = os.path.join(model_base_path_for_gender, "gender_net.caffemodel")
faceNet = cv2.dnn.readNet(faceModel, faceProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Function to highlight faces and detect gender
def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, faceBoxes

# Open webcam
webcam = cv2.VideoCapture(0)
detector = HandGestureDetector()

# Main loop
while webcam.isOpened():
    success, frame = webcam.read()
    if not success:
        print("Failed to capture video frame.")
        break

    # Detect hands and SOS gesture
    frame = detector.run(frame)

    # Highlight faces and detect gender
    resultImg, faceBoxes = highlightFace(faceNet, frame)
    male_count, female_count, violenceByMen, manyMen, loneWomen = 0, 0, False, False, False

    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1]):min(faceBox[3], frame.shape[0]), max(0, faceBox[0]):min(faceBox[2], frame.shape[1])]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.426, 87.769, 114.897), swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderPreds[0].argmax()

        gender_label = "Male" if gender == 0 else "Female"
        if gender == 0:
            male_count += 1
        else:
            female_count += 1

        cv2.putText(resultImg, f'Gender: {gender_label}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if gender == 0:  # Male
            face_crop = cv2.resize(face, (96, 96))
            face_crop = face_crop.astype("float") / 255.0
            face_crop = img_to_array(face_crop)
            face_crop = np.expand_dims(face_crop, axis=0)

            conf = violence_model.predict(face_crop)[0]
            idx = np.argmax(conf)
            label = classes[idx]
            cv2.putText(resultImg, f'{label}: {conf[idx] * 100:.2f}%', (faceBox[0], faceBox[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if label == "violent" and gender == 0:
                violenceByMen = True

    # Additional conditions
    if male_count >= 2 and female_count < male_count:
        manyMen = True
    elif female_count == 1 and male_count == 0:
        loneWomen = True

    # Display warnings
    warning_message = ""
    if violenceByMen:
        warning_message = "WARNING: Violence Detected!"
    elif manyMen:
        warning_message = "WARNING: Many Men Surrounding a Woman!"
    elif loneWomen:
        warning_message = "WARNING: Lone Woman Detected!"

    cv2.putText(resultImg, f'Male: {male_count}, Female: {female_count}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display warning messages on the screen based on conditions
    if warning_message:
        cv2.putText(resultImg, warning_message, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the output frame
    cv2.imshow('Women Safety System - Gesture and Violence Detection', resultImg)

    # Press 'q' to exit the loop and close the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows (this should be outside the loop)
webcam.release()
cv2.destroyAllWindows()



# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array
# import cvlib as cv
# import mediapipe as mp
# import os

# from Help_Detector.DetectHelp import HandGestureDetector
# #from hand_gesture_detector import HandGestureDetector

# punchFlag = False
# manyMen = False
# loneWomen = False
# violenceByMen = False  



# # Load the violence detection model
# violence_model = load_model('TechTitans-Its-A-Hack/Wivsafe_Detection/Violence_Detector/violence_detection.keras')

# # Load the gender detection models
# model_base_path_for_gender = "TechTitans-Its-A-Hack/Wivsafe_Detection/Gender_Detector/"
# faceProto = os.path.join(model_base_path_for_gender, "opencv_face_detector.pbtxt")
# faceModel = os.path.join(model_base_path_for_gender, "opencv_face_detector_uint8.pb")
# ageProto = os.path.join(model_base_path_for_gender, "age_deploy.prototxt")
# ageModel = os.path.join(model_base_path_for_gender, "age_net.caffemodel")
# genderProto = os.path.join(model_base_path_for_gender, "gender_deploy.prototxt")
# genderModel = os.path.join(model_base_path_for_gender, "gender_net.caffemodel")

# # Load face, age, and gender detection models
# faceNet = cv2.dnn.readNet(faceModel, faceProto)
# ageNet = cv2.dnn.readNet(ageModel, ageProto)
# genderNet = cv2.dnn.readNet(genderModel, genderProto)

# # Initialize Mediapipe for hand detection
# mp_drawing = mp.solutions.drawing_utils
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# # Open webcam
# webcam = cv2.VideoCapture(0)

# classes = ['non-violent', 'violent']
# action_state = None
# action_timer = 0
# action_reset_threshold = 30

# # Function to highlight faces
# def highlightFace(net, frame, conf_threshold=0.7):
#     frameOpencvDnn = frame.copy()
#     frameHeight = frameOpencvDnn.shape[0]
#     frameWidth = frameOpencvDnn.shape[1]
#     blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

#     net.setInput(blob)
#     detections = net.forward()
#     faceBoxes = []
#     for i in range(detections.shape[2]):
#         confidence = detections[0, 0, i, 2]
#         if confidence > conf_threshold:
#             x1 = int(detections[0, 0, i, 3] * frameWidth)
#             y1 = int(detections[0, 0, i, 4] * frameHeight)
#             x2 = int(detections[0, 0, i, 5] * frameWidth)
#             y2 = int(detections[0, 0, i, 6] * frameHeight)
#             faceBoxes.append([x1, y1, x2, y2])
#             cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
#     return frameOpencvDnn, faceBoxes

# # Main loop
# while webcam.isOpened():
#     # Read frame from webcam 
#     status, frame = webcam.read()
#     if not status:
#         break  # Exit if no frame is captured

#     # Detect hands
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     hand_results = hands.process(frame_rgb)


#    # Initialize variables for previous finger position
#     prev_index_finger_tip_coords = None
#     punchFlag = False  # Reset flag each frame

#     if hand_results.multi_hand_landmarks:
#         for hand_landmarks in hand_results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#             action_state = "Hands Detected"  # Simple detection for hands

#             # Extract landmark coordinates for the index finger tip and wrist
#             landmarks = hand_landmarks.landmark
#             index_finger_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
#             wrist = landmarks[mp_hands.HandLandmark.WRIST]

#             # Convert to pixel coordinates
#             h, w, _ = frame.shape
#             index_finger_tip_coords = (int(index_finger_tip.x * w), int(index_finger_tip.y * h))
#             wrist_coords = (int(wrist.x * w), int(wrist.y * h))

#             # Calculate the distance between wrist and index finger tip
#             distance = np.linalg.norm(np.array(index_finger_tip_coords) - np.array(wrist_coords))

#             # Define thresholds
#             punch_threshold = 50  # Adjust based on testing
#             speed_threshold = 20   # Adjust based on testing

#             # Detect punching/slapping action based on distance
#             if prev_index_finger_tip_coords is not None:
#                 # Calculate the speed of movement
#                 dx = index_finger_tip_coords[0] - prev_index_finger_tip_coords[0]
#                 dy = index_finger_tip_coords[1] - prev_index_finger_tip_coords[1]
#                 speed = np.linalg.norm(np.array([dx, dy]))

#                 # Check if the movement speed exceeds the threshold and is within punch distance
#                 if speed > speed_threshold and distance < punch_threshold:
#                     punchFlag = True

#                     # Display the punch/slap detection message at the bottom of the camera window
#                     text = "Punch/Slap Detected!"
#                     text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
#                     text_x = (frame.shape[1] - text_size[0]) // 2  # Center the text horizontally
#                     text_y = frame.shape[0] - 20  # Position text 20 pixels from the bottom
#                     cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#             # Update previous position
#             prev_index_finger_tip_coords = index_finger_tip_coords



#     # Highlight faces and detect gender and age
#     resultImg, faceBoxes = highlightFace(faceNet, frame)
#     if not faceBoxes:
#         print("No face detected")
    
#     male_count = 0
#     female_count = 0

#     for faceBox in faceBoxes:
#         face = frame[max(0, faceBox[1]):min(faceBox[3], frame.shape[0]), max(0, faceBox[0]):min(faceBox[2], frame.shape[1])]
        
#         # Preprocess face for gender detection
#         blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.426, 87.769, 114.897), swapRB=False)
        
#         # Gender detection
#         genderNet.setInput(blob)
#         genderPreds = genderNet.forward()
#         gender = genderPreds[0].argmax()
        
#         if gender == 0:  # 0 for Male
#             male_count += 1
#             gender_label = "Male"
#         else:  # 1 for Female
#             female_count += 1
#             gender_label = "Female"

#         # Display gender label above the face box
#         cv2.putText(resultImg, f'Gender: {gender_label}', (faceBox[0], faceBox[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

#         # Detect violence on women
#         if gender_label == "male":
#             # Preprocess face for violence detection
#             face_crop = cv2.resize(face, (96, 96))
#             face_crop = face_crop.astype("float") / 255.0
#             face_crop = img_to_array(face_crop)
#             face_crop = np.expand_dims(face_crop, axis=0)

#             # Apply violence detection on face
#             conf = violence_model.predict(face_crop)[0]
#             idx = np.argmax(conf)
#             label = classes[idx]
#             label = "{}: {:.2f}%".format(label, conf[idx] * 100)

#             Y = faceBox[1] - 10 if faceBox[1] - 10 > 10 else faceBox[1] + 10
#             cv2.putText(resultImg, label, (faceBox[0], Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#             # Set flags based on conditions
#             if label.startswith("violent") and gender == "male":
#                 violenceByMen = True  # Detect if women are in danger
           
#     # Additional checks for multiple men surrounding a woman
#     if male_count >= 2 and female_count < male_count:
#         manyMen = True  # More than 2 men around a woman

#       # Include the flag in further checks
#     if punchFlag and gender == "male":
#         violenceByMen = True  # Or any additional logic you want to apply

#     # Display counts with warning
#     warning_message = ""
#     if female_count==0:
#         warning_message = ""
#     elif violenceByMen:
#         warning_message = "WARNING: Potential Violence Detected!"
#     elif manyMen:
#         warning_message = "WARNING: Many Men Surrounding a Woman!"
#     elif female_count >= 0:
#         loneWomen = True  # A lone woman without any men
#         warning_message = "WARNING: Lone Woman Detected!"
    
#     cv2.putText(resultImg, f'Male: {male_count}, Female: {female_count}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
#     if warning_message:
#         cv2.putText(resultImg, warning_message, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

#     # Display the frame
#     cv2.imshow("Wivsafe-Detection", resultImg)

#     # Break the loop on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

#     # # Create an instance of the HandGestureDetector
#     # detector = HandGestureDetector()

#     # # Run the gesture detection
#     # detector.run()


# # Release the webcam and close windows
# webcam.release()
# cv2.destroyAllWindows()