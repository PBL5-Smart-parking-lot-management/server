from flask import Flask, Response, request, jsonify, send_file
from flask_cors import CORS, cross_origin
import numpy as np
import cv2
import os
from datetime import datetime
from DetectingPlate import is_license_plate, return_crop_img, detect_license_plate

app = Flask(__name__)
CORS(app)

video_data = b''
plate_detected = False  # State to indicate if a plate has been detected

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
    response = Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

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
    try:
        image_data = request.files['image'].read()
        if not os.path.exists('captured_images'):
            os.makedirs('captured_images')
        filename = f'captured_images/image_{len(os.listdir("captured_images"))}.jpg'
        with open(filename, 'wb') as f:
            f.write(image_data)

        result = detect_license_plate(filename)
        print(result)
        if result == "No license plate detected.":
            return jsonify(result="No license plate detected.")

        # Cắt ảnh biển số xe và lưu vào tệp temp_cropped_plate.jpg
        cropped_plate_image = return_crop_img(filename)
        if cropped_plate_image is not None:
            cv2.imwrite('temp_cropped_plate.jpg', cropped_plate_image)

        return jsonify(result=result)
    except Exception as e:
        print("Error saving image:", e)
        return jsonify(result="Error saving image")


@app.route('/plate_image')
def plate_image():
    try:
        temp_plate_image_path = 'temp_cropped_plate.jpg'
        if os.path.exists(temp_plate_image_path):
            print("SUCCESS")
            return send_file(temp_plate_image_path, mimetype='image/jpeg')
        else:
            print("FAIL")
            return "No plate image available"
    except Exception as e:
        print("Error processing plate image:", e)
        return "Error processing plate image"


if __name__ == '__main__':
    if not os.path.exists('captured_images'):
        os.makedirs('captured_images')
    app.run(host='0.0.0.0', port=5000)
