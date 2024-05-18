from flask import Flask, Response, request
from flask_cors import CORS, cross_origin
import numpy as np
import cv2
import os

app = Flask(__name__)
CORS(app)

video_data = b''

@app.route('/upload', methods=['POST'])
def upload():
    global video_data
    video_data = request.data
    return "OK"

def generate_frames():
    global video_data
    while True:
        if video_data:
            nparr = np.frombuffer(video_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    yield (b'--frame\r\n' 
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            else:
                print("Error decoding frame")

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['GET'])
def capture():
    global video_data
    nparr = np.frombuffer(video_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is not None:
        _, buffer = cv2.imencode('.jpg', frame)
        return Response(buffer.tobytes(), mimetype='image/jpeg')
    return "No video data", 400

@app.route('/save_image', methods=['POST'])
@cross_origin()
def save_image():
    image_data = request.files['image'].read()
    if not os.path.exists('captured_images'):
        os.makedirs('captured_images')
    filename = f'captured_images/image_{len(os.listdir("captured_images"))}.jpg'
    with open(filename, 'wb') as f:
        f.write(image_data)
    return 'Ảnh đã được lưu thành công!'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
