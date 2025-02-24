import cv2
import argparse
import numpy as np
import os

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

def detectHands(frame, faceBox):
    hands = []
    face_height = faceBox[3] - faceBox[1]
    y_start = max(0, faceBox[1] - int(1.5 * face_height))
    y_end = faceBox[1]
    hands.append((faceBox[0], y_start, faceBox[2], y_end))
    return hands

def checkSOSGesture(faceBox, hands):
    for hand in hands:
        if hand[1] < faceBox[1]:
            return True
    return False

# Create an argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--image')
args = parser.parse_args()

# Define the paths for the models and prototxt files
model_base_path = "Wivsafe_Detection/"

# Update the file paths to the correct absolute paths
faceProto = os.path.join(model_base_path, "opencv_face_detector.pbtxt")
faceModel = os.path.join(model_base_path, "opencv_face_detector_uint8.pb")
ageProto = os.path.join(model_base_path, "age_deploy.prototxt")
ageModel = os.path.join(model_base_path, "age_net.caffemodel")
genderProto = os.path.join(model_base_path, "gender_deploy.prototxt")
genderModel = os.path.join(model_base_path, "gender_net.caffemodel")

# Define the mean values for the model
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load the models
try:
    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)
except cv2.error as e:
    print(f"Error loading model files: {e}")
    exit(1)

video = cv2.VideoCapture(args.image if args.image else 0)
padding = 20

male_count = 0
female_count = 0

while cv2.waitKey(1) < 0:
    hasFrame, frame = video.read()
    if not hasFrame:
        cv2.waitKey()
        break

    resultImg, faceBoxes = highlightFace(faceNet, frame)
    if not faceBoxes:
        print("No face detected")

    male_count = 0  
    female_count = 0

    genders = []
    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1] - padding): min(faceBox[3] + padding, frame.shape[0] - 1), max(0, faceBox[0] - padding): min(faceBox[2] + padding, frame.shape[1] - 1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        genders.append(gender)

        if gender == 'Male':
            male_count += 1
        else:
            female_count += 1

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        print(f'Gender: {gender}, Age: {age[1:-1]} years')

        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        # Detect hands near the face
        hands = detectHands(frame, faceBox)
        for hand in hands:
            cv2.rectangle(resultImg, (hand[0], hand[1]), (hand[2], hand[3]), (255, 0, 0), 2)

        # Only check for SOS gesture if a woman is detected
        if gender == 'Female' and checkSOSGesture(faceBox, hands):
            cv2.putText(resultImg, "SOS Detected!", (faceBox[0], faceBox[1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the gender counts on the frame
    cv2.putText(resultImg, f'Male: {male_count}, Female: {female_count}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow("Detecting age and gender", resultImg)
