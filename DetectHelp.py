import cv2
import time
import HandTracking as htm
from location import LocationService  # Import location service
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import Integer, String, Float

app = Flask(__name__)

##CREATE DATABASE
class Base(DeclarativeBase):
    pass

# configure the SQLite database, relative to the app instance folder
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///newCoordinates123.db"

db = SQLAlchemy(model_class=Base)
# initialize the app with the extension
db.init_app(app)

class Coord(db.Model):
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    state: Mapped[str] = mapped_column(String, nullable=False)
    address: Mapped[str] = mapped_column(String, nullable=False)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)  # Corrected column name
    longitude: Mapped[float] = mapped_column(Float, nullable=False)  # Corrected column name

    def __repr__(self):
        return f'<Coord {self.state}, {self.address}>'
   
# creates the table
with app.app_context():
    db.create_all()


class HandGestureDetector:
    def __init__(self, wCam=640, hCam=480, detectionCon=0.75):
        # Set camera width and height
        self.wCam = wCam
        self.hCam = hCam

        # Initialize the camera
        self.cap = cv2.VideoCapture(0)

        # Check if the camera opened successfully
        if not self.cap.isOpened():
            print("Error: Camera failed to initialize.")
            exit()

        self.cap.set(3, self.wCam)
        self.cap.set(4, self.hCam)

        # Initialize hand detector
        self.detector = htm.handDetector(detectionCon=detectionCon)
        self.tipIds = [4, 8, 12, 16, 20]

        # Frame timing variables for FPS calculation
        self.pTime = 0

        # List to track the sequence of detected gestures
        self.gesture_sequence = []

        # Variables for displaying "HELP"
        self.help_detected = False
        self.help_start_time = 0
        self.help_duration = 3  # Duration for which HELP is displayed (in seconds)
        self.flash_interval = 0.5  # Flash interval in seconds
        self.last_flash_time = 0
        self.show_help = True  # Toggle for flashing HELP

        # Initialize location service
        self.location_service = LocationService()

    def is_help_sign(self, lmList):
        """Detect if the hand gesture corresponds to one of the 'help' gestures."""
        if len(lmList) == 0:
            return None

        # Thumb check
        thumb_up = lmList[self.tipIds[0]][1] > lmList[self.tipIds[0] - 1][1]  # Thumb straight
        
        # Other fingers check
        fingers_up = []
        for id in range(1, 5):
            fingers_up.append(lmList[self.tipIds[id]][2] < lmList[self.tipIds[id] - 2][2])  # Finger straight

        # Conditions for detecting each 'help' gesture
        if all(fingers_up) and thumb_up:
            return "Open Hand"
        elif not thumb_up and all(fingers_up):
            return "Thumb Curled"
        elif not any(fingers_up):
            return "Fist"
        
        return None

    def check_for_help_sequence(self):
        """Check if the sequence of gestures corresponds to 'help': Open Hand -> Thumb Curled -> Fist."""
        return self.gesture_sequence == ["Open Hand", "Thumb Curled", "Fist"]

    def run(self):
        """Run the hand gesture detection loop."""
        while True:
            # Capture image from the camera
            success, img = self.cap.read()
            if not success:
                print("Error: Failed to capture image from camera.")
                break

            # Detect hands and landmarks
            img = self.detector.findHands(img)
            lmList = self.detector.findPosition(img, draw=False)

            if len(lmList) != 0:
                gesture = self.is_help_sign(lmList)
                
                if gesture:
                    # Add the gesture to the sequence
                    if len(self.gesture_sequence) == 0 or gesture != self.gesture_sequence[-1]:
                        self.gesture_sequence.append(gesture)

                    # Only keep the last 3 gestures in the sequence
                    if len(self.gesture_sequence) > 3:
                        self.gesture_sequence.pop(0)

                    # Check for "HELP" sequence
                    if self.check_for_help_sequence():
                        self.help_detected = True
                        self.help_start_time = time.time()  # Set the start time for displaying HELP
                        self.gesture_sequence.clear()  # Clear sequence after detection
                        
                        # Fetch and display location
                        self.location_service.get_location_and_display() 
                        lat_long = self.location_service.get_current_location_details()
                        if lat_long:
                            state = lat_long['state']
                            address = lat_long['address']
                            coordinates = lat_long['coordinates']
                            latitude = coordinates[0] if coordinates else 0.0
                            longitude = coordinates[1] if coordinates else 0.0

                    # Save data to the database
                            new_data = Coord(state=state, address=address, latitude=latitude,longitude=longitude)
                            with app.app_context():  # Make sure you're within the app context
                                db.session.add(new_data)
                                db.session.commit()

                            print("Data saved to the database.")  # Call your location service
                        print("HELP detected!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")  # Print when help is detected

                # Display the detected gesture on the image
                if gesture:
                    cv2.putText(img, gesture, (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

            # Check if we should display "HELP"
            if self.help_detected:
                elapsed_time = time.time() - self.help_start_time

                # Flash HELP every `flash_interval` seconds
                if time.time() - self.last_flash_time >= self.flash_interval:
                    self.show_help = not self.show_help  # Toggle the flash effect
                    self.last_flash_time = time.time()

                if elapsed_time <= self.help_duration:
                    if self.show_help:
                        cv2.putText(img, "HELP", (200, 400), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 5)
                else:
                    self.help_detected = False  # Stop displaying HELP after the duration

            # Calculate and display FPS
            cTime = time.time()
            fps = 1 / (cTime - self.pTime)
            self.pTime = cTime
            cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

            # Show the image
            cv2.imshow("Image", img)

            # Press 'q' to exit the loop and close the window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = HandGestureDetector()
    detector.run()
