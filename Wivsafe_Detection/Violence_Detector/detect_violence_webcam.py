import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cvlib as cv
import mediapipe as mp

# Load model
model = load_model(r'Wivsafe_Detection\Violence_Detector\violence_detection.keras')

# Initialize Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Open webcam
webcam = cv2.VideoCapture(0)

classes = ['non-violent', 'violent']

# Action state tracking
action_state = None
action_timer = 0  # Timer to track frames since last action
action_reset_threshold = 30  # Number of frames to wait before resetting action

# Loop through frames
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while webcam.isOpened():
        # Read frame from webcam 
        status, frame = webcam.read()
        if not status:
            break  # Exit if no frame is captured

        # Convert the BGR image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        # Detect hands and draw landmarks
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Analyze hand positions for action detection
                hand_positions = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                index_finger_y = hand_positions[mp_hands.HandLandmark.INDEX_FINGER_TIP.value][1]
                wrist_y = hand_positions[mp_hands.HandLandmark.WRIST.value][1]

                # Update action state based on position
                if index_finger_y < wrist_y:
                    action_state = "Punching"
                else:
                    # Check for grabbing by comparing thumb and index finger positions
                    thumb_tip = hand_positions[mp_hands.HandLandmark.THUMB_TIP.value]
                    index_tip = hand_positions[mp_hands.HandLandmark.INDEX_FINGER_TIP.value]

                    # Calculate the distance between thumb and index finger tips
                    distance = np.linalg.norm(np.array(thumb_tip) - np.array(index_tip))
                    
                    # Define a threshold for grabbing
                    grabbing_threshold = 0.05  # Adjust this threshold as needed

                    if distance < grabbing_threshold:
                        action_state = "Grabbing"
                    else:
                        action_state = None

        # Apply face detection
        face, confidence = cv.detect_face(frame)

        # Loop through detected faces
        for idx, f in enumerate(face):
            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]

            # Draw rectangle over face
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # Crop the detected face region
            face_crop = np.copy(frame[startY:endY, startX:endX])

            if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
                continue

            # Preprocessing for violence detection model
            face_crop = cv2.resize(face_crop, (96, 96))
            face_crop = face_crop.astype("float") / 255.0
            face_crop = img_to_array(face_crop)
            face_crop = np.expand_dims(face_crop, axis=0)

            # Apply violence detection on face
            conf = model.predict(face_crop)[0]
            idx = np.argmax(conf)
            label = classes[idx]
            label = "{}: {:.2f}%".format(label, conf[idx] * 100)

            Y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Only display the action label if a violent action is detected
        if action_state is not None:
            cv2.putText(frame, f"Violent | Action: {action_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            action_timer = 0  # Reset timer when an action is detected
        else:
            # Increment the timer if no violent action is detected
            action_timer += 1

            # Reset action state to None if the timer exceeds threshold
            if action_timer > action_reset_threshold:
                action_state = None  # Set action state to None when no action is detected

        # Display the frame
        cv2.imshow("Webcam", frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close windows
webcam.release()
cv2.destroyAllWindows()



# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array
# import cvlib as cv
# import mediapipe as mp

# # Load model
# model = load_model(r'C:\Users\junai\Desktop\Women in danger data set\Violence-Detector\violence_detection.keras')

# # Initialize Mediapipe
# mp_drawing = mp.solutions.drawing_utils
# mp_hands = mp.solutions.hands

# # Open webcam
# webcam = cv2.VideoCapture(0)

# classes = ['non-violent', 'violent']

# # Loop through frames
# with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
#     while webcam.isOpened():
#         # Read frame from webcam 
#         status, frame = webcam.read()
#         if not status:
#             break  # Exit if no frame is captured

#         # Convert the BGR image to RGB
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         result = hands.process(frame_rgb)

#         # Detect hands and draw landmarks
#         if result.multi_hand_landmarks:
#             for hand_landmarks in result.multi_hand_landmarks:
#                 mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#                 # Example: Check for punching or grabbing movements
#                 # Here you would add your logic to analyze the hand positions
#                 # For simplicity, we'll just add a placeholder for actions
#                 hand_positions = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
#                 # Simple action detection (placeholder logic)
#                 if hand_positions[mp_hands.HandLandmark.INDEX_FINGER_TIP.value][1] < hand_positions[mp_hands.HandLandmark.WRIST.value][1]:
#                     action_label = "Punching"
#                 else:
#                     action_label = "Neutral"

#                 # Put action label on frame
#                 cv2.putText(frame, action_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

#         # Apply face detection
#         face, confidence = cv.detect_face(frame)

#         # Loop through detected faces
#         for idx, f in enumerate(face):
#             (startX, startY) = f[0], f[1]
#             (endX, endY) = f[2], f[3]

#             # Draw rectangle over face
#             cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

#             # Crop the detected face region
#             face_crop = np.copy(frame[startY:endY, startX:endX])

#             if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
#                 continue

#             # Preprocessing for violence detection model
#             face_crop = cv2.resize(face_crop, (96, 96))
#             face_crop = face_crop.astype("float") / 255.0
#             face_crop = img_to_array(face_crop)
#             face_crop = np.expand_dims(face_crop, axis=0)

#             # Apply violence detection on face
#             conf = model.predict(face_crop)[0]
#             idx = np.argmax(conf)
#             label = classes[idx]
#             label = "{}: {:.2f}%".format(label, conf[idx] * 100)

#             Y = startY - 10 if startY - 10 > 10 else startY + 10
#             cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#         # Display the frame
#         cv2.imshow("Webcam", frame)

#         # Break the loop on 'q' key press
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# # Release the webcam and close windows
# webcam.release()
# cv2.destroyAllWindows()




# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array
# import cvlib as cv

# # Load model
# model = load_model(r'C:\Users\junai\Desktop\Women in danger data set\Violence-Detector\violence_detection.keras')

# # Open webcam
# webcam = cv2.VideoCapture(0)

# classes = ['non-violent', 'violent']

# # Loop through frames
# while webcam.isOpened():
#     # Read frame from webcam 
#     status, frame = webcam.read()
#     if not status:
#         break  # Exit if no frame is captured

#     # Apply face detection
#     face, confidence = cv.detect_face(frame)

#     # Loop through detected faces
#     for idx, f in enumerate(face):
#         (startX, startY) = f[0], f[1]
#         (endX, endY) = f[2], f[3]

#         # Draw rectangle over face
#         cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

#         # Crop the detected face region
#         face_crop = np.copy(frame[startY:endY, startX:endX])

#         if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
#             continue

#         # Preprocessing for violence detection model
#         face_crop = cv2.resize(face_crop, (96, 96))
#         face_crop = face_crop.astype("float") / 255.0
#         face_crop = img_to_array(face_crop)
#         face_crop = np.expand_dims(face_crop, axis=0)

#         # Apply violence detection on face
#         conf = model.predict(face_crop)[0]
#         idx = np.argmax(conf)
#         label = classes[idx]
#         label = "{}: {:.2f}%".format(label, conf[idx] * 100)

#         Y = startY - 10 if startY - 10 > 10 else startY + 10
#         cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#     # Display the frame
#     cv2.imshow("Webcam", frame)

#     # Break the loop on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the webcam and close windows
# webcam.release()
# cv2.destroyAllWindows()
