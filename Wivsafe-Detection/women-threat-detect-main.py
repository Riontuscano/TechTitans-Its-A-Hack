import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cvlib as cv
import mediapipe as mp
import os

punchFlag = False
manyMen = False
loneWomen = False
violenceByMen = False  # punch, slap, hit on face, beating 

# Load the violence detection model
violence_model = load_model('Wivsafe-Detection/Violence-Detector/violence_detection.keras')

# Load the gender detection models
model_base_path_for_gender = "Wivsafe-Detection/Gender-Detector/"
faceProto = os.path.join(model_base_path_for_gender, "opencv_face_detector.pbtxt")
faceModel = os.path.join(model_base_path_for_gender, "opencv_face_detector_uint8.pb")
ageProto = os.path.join(model_base_path_for_gender, "age_deploy.prototxt")
ageModel = os.path.join(model_base_path_for_gender, "age_net.caffemodel")
genderProto = os.path.join(model_base_path_for_gender, "gender_deploy.prototxt")
genderModel = os.path.join(model_base_path_for_gender, "gender_net.caffemodel")

# Load face, age, and gender detection models
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Initialize Mediapipe for hand detection
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Open webcam
webcam = cv2.VideoCapture(0)

classes = ['non-violent', 'violent']
action_state = None
action_timer = 0
action_reset_threshold = 30

# Function to highlight faces
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

# Main loop
while webcam.isOpened():
    # Read frame from webcam 
    status, frame = webcam.read()
    if not status:
        break  # Exit if no frame is captured

    # Detect hands
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(frame_rgb)


   # Initialize variables for previous finger position
    prev_index_finger_tip_coords = None
    punchFlag = False  # Reset flag each frame

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            action_state = "Hands Detected"  # Simple detection for hands

            # Extract landmark coordinates for the index finger tip and wrist
            landmarks = hand_landmarks.landmark
            index_finger_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            wrist = landmarks[mp_hands.HandLandmark.WRIST]

            # Convert to pixel coordinates
            h, w, _ = frame.shape
            index_finger_tip_coords = (int(index_finger_tip.x * w), int(index_finger_tip.y * h))
            wrist_coords = (int(wrist.x * w), int(wrist.y * h))

            # Calculate the distance between wrist and index finger tip
            distance = np.linalg.norm(np.array(index_finger_tip_coords) - np.array(wrist_coords))

            # Define thresholds
            punch_threshold = 50  # Adjust based on testing
            speed_threshold = 20   # Adjust based on testing

            # Detect punching/slapping action based on distance
            if prev_index_finger_tip_coords is not None:
                # Calculate the speed of movement
                dx = index_finger_tip_coords[0] - prev_index_finger_tip_coords[0]
                dy = index_finger_tip_coords[1] - prev_index_finger_tip_coords[1]
                speed = np.linalg.norm(np.array([dx, dy]))

                # Check if the movement speed exceeds the threshold and is within punch distance
                if speed > speed_threshold and distance < punch_threshold:
                    punchFlag = True

                    # Display the punch/slap detection message at the bottom of the camera window
                    text = "Punch/Slap Detected!"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                    text_x = (frame.shape[1] - text_size[0]) // 2  # Center the text horizontally
                    text_y = frame.shape[0] - 20  # Position text 20 pixels from the bottom
                    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Update previous position
            prev_index_finger_tip_coords = index_finger_tip_coords



    # Highlight faces and detect gender and age
    resultImg, faceBoxes = highlightFace(faceNet, frame)
    if not faceBoxes:
        print("No face detected")
    
    male_count = 0
    female_count = 0

    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1]):min(faceBox[3], frame.shape[0]), max(0, faceBox[0]):min(faceBox[2], frame.shape[1])]
        
        # Preprocess face for gender detection
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.426, 87.769, 114.897), swapRB=False)
        
        # Gender detection
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderPreds[0].argmax()
        
        if gender == 0:  # 0 for Male
            male_count += 1
            gender_label = "Male"
        else:  # 1 for Female
            female_count += 1
            gender_label = "Female"

        # Display gender label above the face box
        cv2.putText(resultImg, f'Gender: {gender_label}', (faceBox[0], faceBox[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        # Detect violence on women
        if gender_label == "male":
            # Preprocess face for violence detection
            face_crop = cv2.resize(face, (96, 96))
            face_crop = face_crop.astype("float") / 255.0
            face_crop = img_to_array(face_crop)
            face_crop = np.expand_dims(face_crop, axis=0)

            # Apply violence detection on face
            conf = violence_model.predict(face_crop)[0]
            idx = np.argmax(conf)
            label = classes[idx]
            label = "{}: {:.2f}%".format(label, conf[idx] * 100)

            Y = faceBox[1] - 10 if faceBox[1] - 10 > 10 else faceBox[1] + 10
            cv2.putText(resultImg, label, (faceBox[0], Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Set flags based on conditions
            if label.startswith("violent") and gender == "male":
                violenceByMen = True  # Detect if women are in danger
           
    # Additional checks for multiple men surrounding a woman
    if male_count >= 2 and female_count < male_count:
        manyMen = True  # More than 2 men around a woman

      # Include the flag in further checks
    if punchFlag and gender == "male":
        violenceByMen = True  # Or any additional logic you want to apply

    # Display counts with warning
    warning_message = ""
    if female_count==0:
        warning_message = ""
    elif violenceByMen:
        warning_message = "WARNING: Potential Violence Detected!"
    elif manyMen:
        warning_message = "WARNING: Many Men Surrounding a Woman!"
    elif female_count >= 0:
        loneWomen = True  # A lone woman without any men
        warning_message = "WARNING: Lone Woman Detected!"
    
    cv2.putText(resultImg, f'Male: {male_count}, Female: {female_count}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    if warning_message:
        cv2.putText(resultImg, warning_message, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow("Wivsafe-Detection", resultImg)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
webcam.release()
cv2.destroyAllWindows()