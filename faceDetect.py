from flask import Flask, render_template, Response
from flask_opencv_streamer.streamer import Streamer
import cv2
import threading
from deepface import DeepFace

app = Flask(__name__, static_url_path='/static')

# Initialize global variables
face_match = False
frame = None

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
reference_img = cv2.imread("reference.png")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize Streamer
port = 8000
require_login = False
streamer = Streamer(port, require_login)

# Function to check face in a separate thread
def check_face(frame):
    global face_match
    try:
        result = DeepFace.verify(frame, reference_img)
        face_match = result['verified']
    except Exception as e:
        print(f"Error in face verification: {e}")
        face_match = False

# Function to capture video and update frame
def capture_video():
    global frame
    counter = 0
    start_point = (200, 100)
    end_point = (450, 350)
    thickness = 3

    while True:
        ret, frame = cap.read()
        if ret:
            if counter % 30 == 0:
                # Launch a new thread for face verification
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            counter += 1

            if face_match:
                cv2.putText(frame, "Face Matched!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(frame, start_point, end_point, (0, 255, 0), thickness)
            else:
                cv2.putText(frame, "Face Not Matched!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.rectangle(frame, start_point, end_point, (0, 0, 255), thickness)
            
            streamer.update_frame(frame)

            if not streamer.is_streaming:
                streamer.start_streaming()

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('app.html')

@app.route('/video_feed')
def video_feed():
    return Response(streamer.generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    video_thread = threading.Thread(target=capture_video, daemon=True)
    video_thread.start()
    app.run(host='0.0.0.0', port=port, debug=True)
