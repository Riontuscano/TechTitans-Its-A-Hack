import cv2
import time
from flask import Flask, render_template, Response
from DetectHelp import HandGestureDetector  # Import your existing HandGestureDetector class
from location import LocationService  # Import location service

app = Flask(__name__)

class GestureApp:
    def __init__(self):
        self.detector = HandGestureDetector()  # Initialize the gesture detector
        self.location_service = LocationService()  # Initialize location service

    def generate_frames(self):
        while True:
            success, img = self.detector.cap.read()
            if not success:
                print("Failed to capture frame.")
                break

            # Detect hands and landmarks
            img = self.detector.detector.findHands(img)
            lmList = self.detector.detector.findPosition(img, draw=False)

            if len(lmList) != 0:
                gesture = self.detector.is_help_sign(lmList)

                if gesture:
                    # Add the gesture to the sequence
                    if len(self.detector.gesture_sequence) == 0 or gesture != self.detector.gesture_sequence[-1]:
                        self.detector.gesture_sequence.append(gesture)

                    # Only keep the last 3 gestures in the sequence
                    if len(self.detector.gesture_sequence) > 3:
                        self.detector.gesture_sequence.pop(0)

                    # Check for "HELP" sequence
                    if self.detector.check_for_help_sequence():
                        self.detector.help_detected = True
                        self.detector.help_start_time = time.time()  # Set the start time for displaying HELP
                        self.detector.gesture_sequence.clear()  # Clear sequence after detection
                        self.location_service.get_location_and_display()  # Call your location service
                        print("HELP detected!")  # Print when help is detected

                # Display the detected gesture on the image
                if gesture:
                    cv2.putText(img, gesture, (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

            # Check if we should display "HELP"
            if self.detector.help_detected:
                elapsed_time = time.time() - self.detector.help_start_time

                # Flash HELP every `flash_interval` seconds
                if time.time() - self.detector.last_flash_time >= self.detector.flash_interval:
                    self.detector.show_help = not self.detector.show_help  # Toggle the flash effect
                    self.detector.last_flash_time = time.time()

                if elapsed_time <= self.detector.help_duration:
                    if self.detector.show_help:
                        cv2.putText(img, "HELP", (200, 400), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 5)
                else:
                    self.detector.help_detected = False  # Stop displaying HELP after the duration

            # Convert the frame to JPEG format for streaming
            _, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/CCtv')
def CCtv():
    return render_template('CCtv.html')

@app.route('/video_feed')
def video_feed():
    gesture_app = GestureApp()  # Create a new instance of GestureApp
    return Response(gesture_app.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
