import cv2
from flask import Flask, render_template, Response
from DetectHelp import HandGestureDetector  # Import your gesture detection class

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/CCtv')
def CCtv():
    return render_template('CCtv.html')

def generate_frames():
    detector = HandGestureDetector()

    while True:
        success, frame = detector.cap.read()
        if not success:
            break
        
        # Process the frame through your gesture detection code here
        img = detector.detector.findHands(frame)
        lmList = detector.detector.findPosition(img, draw=False)

        if lmList:
            gesture = detector.is_help_sign(lmList)
            if gesture:
                cv2.putText(img, gesture, (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        # Convert the frame to JPEG format for streaming
        _, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
