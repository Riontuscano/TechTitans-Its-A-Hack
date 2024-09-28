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

while True:
    # Capture image from the camera
    success, img = cap.read()
    if not success:
        print("Error: Failed to capture image from camera.")
        break

    # Detect hands and landmarks
    img = detector.findHands(img)

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
