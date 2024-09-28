import cv2
import time
import HandTracking as htm

# Set camera width and height
wCam, hCam = 640, 480

# Initialize the camera
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Camera failed to initialize.")
    exit()

cap.set(3, wCam)
cap.set(4, hCam)

# Initialize hand detector
detector = htm.handDetector(detectionCon=0.75)

# Frame timing variables for FPS calculation
pTime = 0

# Function to detect specific hand gestures
tipIds = [4, 8, 12, 16, 20]

def is_help_sign(lmList):
    """
    Detects 'Open Hand', 'Thumb Curled', and 'Fist' gestures based on landmarks.
    """
    if len(lmList) == 0:
        return None

    thumb_up = lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]  # Thumb straight
    
    fingers_up = []
    for id in range(1, 5):
        fingers_up.append(lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2])  # Finger straight

    if all(fingers_up) and thumb_up:
        return "Open Hand"
    elif not thumb_up and all(fingers_up):
        return "Thumb Curled"
    elif not any(fingers_up):
        return "Fist"
    
    return None

def check_for_help_sequence(gesture_sequence):
    """
    Check if the sequence of gestures corresponds to 'help': Open Hand -> Thumb Curled -> Fist.
    """
    return gesture_sequence == ["Open Hand", "Thumb Curled", "Fist"]

# List to track the sequence of detected gestures
gesture_sequence = []

# Variables for displaying "HELP"
help_detected = False
help_start_time = 0
help_duration = 3  # Duration for which HELP is displayed (in seconds)
flash_interval = 0.5  # Flash interval in seconds
last_flash_time = 0
show_help = True  # Toggle for flashing HELP

while True:
    # Capture image from the camera
    success, img = cap.read()
    if not success:
        print("Error: Failed to capture image from camera.")
        break

    # Detect hands and landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        gesture = is_help_sign(lmList)
        
        # Add the gesture to the sequence
        if gesture:
            if len(gesture_sequence) == 0 or gesture != gesture_sequence[-1]:
                gesture_sequence.append(gesture)

            # Only keep the last 3 gestures in the sequence
            if len(gesture_sequence) > 3:
                gesture_sequence.pop(0)

            # Check for "HELP" sequence
            if check_for_help_sequence(gesture_sequence):
                help_detected = True
                help_start_time = time.time()  # Set the start time for displaying HELP
                gesture_sequence.clear()  # Clear sequence after detection

        # Display the detected gesture on the image
        if gesture:
            cv2.putText(img, gesture, (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # Check if we should display "HELP" (for the duration defined)
    if help_detected:
        elapsed_time = time.time() - help_start_time

        # Flash HELP every `flash_interval` seconds
        if time.time() - last_flash_time >= flash_interval:
            show_help = not show_help  # Toggle the flash effect
            last_flash_time = time.time()

        # If the time elapsed since "HELP" was detected is within the allowed duration
        if elapsed_time <= help_duration:
            if show_help:
                cv2.putText(img, "HELP", (200, 400), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 5)
        else:
            help_detected = False  # Stop displaying HELP after the duration

    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # Show the image
    cv2.imshow("Image", img)

    # Press 'q' to exit the loop and close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
