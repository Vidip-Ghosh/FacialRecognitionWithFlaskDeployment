import cloudinary.uploader
from flask import Flask, render_template, Response, request, redirect,flash,jsonify,send_file
from flask_opencv_streamer.streamer import Streamer
import cv2
import threading
from deepface import DeepFace
import cloudinary
          
cloudinary.config( 
  cloud_name = "dmfxvx5vn", 
  api_key = "429888616999743", 
  api_secret = "vY_6H1ltWNDXK-vghY3FrUHffpo" 
)

app = Flask(__name__)

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
port = 8001
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

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part') 
        return redirect(request.url)
    file=request.files['file']
    if file.filename=='': 
        flash('No selected file')
        return redirect(request.url)
    if file: 
        # Upload the file to Cloudinary
        result = cloudinary.uploader.upload(file)
        print("Cloudinary Result:", result)
        # return render_template("upload.html",file=file)
        return redirect(result['secure_url']) 

if __name__ == '__main__':
    video_thread = threading.Thread(target=capture_video, daemon=True)
    video_thread.start()
    app.run(host='0.0.0.0', port=port, debug=True)
