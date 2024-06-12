import cv2
import numpy as np
import torch
from detect import detect
from models.experimental import attempt_load
from src.char_classification.model import CNN_Model
from utils_LP import character_recog_CNN, crop_n_rotate_LP

# check có biển số xe và nhận diện kí tự
def detect_license_plate(image_path):
    Min_char = 0.01
    Max_char = 0.09
    CHAR_CLASSIFICATION_WEIGHTS = './src/weights/weight.h5'
    LP_weights = 'LP_detect_yolov7_500img.pt'

    # Tải mô hình nhận diện ký tự
    model_char = CNN_Model(trainable=False).model
    try:
        model_char.load_weights(CHAR_CLASSIFICATION_WEIGHTS)
    except Exception as e:
        return f"Error loading character classification weights: {e}"

    # Xác định thiết bị (GPU hoặc CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tải mô hình nhận diện biển số xe
    try:
        model_LP = attempt_load(LP_weights, map_location=device)
    except Exception as e:
        return f"Error loading license plate detection weights: {e}"

    source_img = cv2.imread(image_path)
    if source_img is None:
        return "Image not found or unable to load."

    try:
        pred, LP_detected_img = detect(model_LP, source_img, device, imgsz=640)
    except Exception as e:
        return "No license plate detected."

    if len(pred) == 0:
        return "No license plate detected."
    else:
        results = []
        for c, (*xyxy, conf, cls) in enumerate(reversed(pred)):
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            try:
                angle, rotate_thresh, LP_rotated = crop_n_rotate_LP(source_img, x1, y1, x2, y2)
            except Exception as e:
                continue

            if rotate_thresh is None or LP_rotated is None:
                continue

            LP_rotated_copy = LP_rotated.copy()
            cont, hier = cv2.findContours(rotate_thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            cont = sorted(cont, key=cv2.contourArea, reverse=True)[:17]

            char_x = []
            height, width, _ = LP_rotated_copy.shape
            roiarea = height * width

            for cnt in cont:
                x, y, w, h = cv2.boundingRect(cnt)
                ratiochar = w / h
                char_area = w * h
                if (Min_char * roiarea < char_area < Max_char * roiarea) and (0.25 < ratiochar < 0.7):
                    char_x.append([x, y, w, h])

            if not char_x:
                continue

            char_x = np.array(char_x)
            threshold_12line = char_x[:, 1].min() + (char_x[:, 3].mean() / 2)
            char_x = sorted(char_x, key=lambda x: x[0], reverse=False)
            first_line = ""
            second_line = ""

            for char in char_x:
                x, y, w, h = char
                imgROI = rotate_thresh[y:y + h, x:x + w]
                try:
                    text = character_recog_CNN(model_char, imgROI)
                except Exception as e:
                    text = 'Background'

                if text == 'Background':
                    text = ''

                if y < threshold_12line:
                    first_line += text
                else:
                    second_line += text

            strFinalString = first_line + second_line
            if strFinalString:
                results.append(strFinalString)
                cv2.putText(LP_detected_img, strFinalString, (x1, y1 - 20), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 2)

        if not results:
            return "No license plate detected."

    return results

# check có biển số xe không = frame
def is_license_plate(frame):
    weights_path = 'LP_detect_yolov7_500img.pt'

    # Xác định thiết bị (GPU hoặc CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Tải mô hình nhận diện biển số xe
    model = attempt_load(weights_path, map_location=device)
    model = model.half() if device.type != 'cpu' else model.float()

    # Thực hiện dự đoán
    try:
        pred, img = detect(model, frame, device)
        print("Predictions:", pred)
    except Exception as e:
        print("Error during detection:", e)
        return False

    # Kiểm tra xem có biển số xe được phát hiện hay không
    license_plate_detected = False
    for i, det in enumerate(pred):
        if len(det):
            license_plate_detected = True
            break

    return license_plate_detected

# check có biển số xe không = image_path
def is_license_plate(image_path):
    weights_path = 'LP_detect_yolov7_500img.pt'
    # Xác định thiết bị (GPU hoặc CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Tải mô hình nhận diện biển số xe
    model = attempt_load(weights_path, map_location=device)
    model = model.half() if device.type != 'cpu' else model.float()

    # Đọc ảnh đầu vào
    img0 = cv2.imread(image_path)
    assert img0 is not None, 'Image Not Found ' + image_path

    # Thực hiện dự đoán
    pred, img = detect(model, img0, device)

    # Kiểm tra xem có biển số xe được phát hiện hay không
    license_plate_detected = False
    for i, det in enumerate(pred):
        if len(det):
            license_plate_detected = True
            break

    return license_plate_detected


def return_crop_img(image_path):
    Min_char = 0.01
    Max_char = 0.09
    CHAR_CLASSIFICATION_WEIGHTS = './src/weights/weight.h5'
    LP_weights = 'LP_detect_yolov7_500img.pt'

    # Tải mô hình nhận diện ký tự
    model_char = CNN_Model(trainable=False).model
    try:
        model_char.load_weights(CHAR_CLASSIFICATION_WEIGHTS)
    except Exception as e:
        return f"Error loading character classification weights: {e}"

    # Xác định thiết bị (GPU hoặc CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tải mô hình nhận diện biển số xe
    try:
        model_LP = attempt_load(LP_weights, map_location=device)
    except Exception as e:
        return f"Error loading license plate detection weights: {e}"

    source_img = cv2.imread(image_path)
    if source_img is None:
        return "Image not found or unable to load."

    try:
        pred, LP_detected_img = detect(model_LP, source_img, device, imgsz=640)
    except Exception as e:
        return "No license plate detected."

    if len(pred) == 0:
        return "No license plate detected."
    else:
        for c, (*xyxy, conf, cls) in enumerate(reversed(pred)):
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            try:
                angle, rotate_thresh, LP_rotated = crop_n_rotate_LP(source_img, x1, y1, x2, y2)
            except Exception as e:
                continue

            if rotate_thresh is None or LP_rotated is None:
                continue

            # Trả về ảnh của biển số sau khi đã được cắt và xoay
            return LP_rotated

    return None  # Trả về None nếu không có biển số xe được phát hiện

if __name__ == "__main__":
    image_path = 'E:/PycharmProjects/License-Plate-Recognition-YOLOv7-and-CNN/image_1.jpg'

    # Gọi hàm return_crop_img để nhận ảnh đã được cắt từ biển số
    cropped_img = return_crop_img(image_path)

    # Kiểm tra xem ảnh đã được trả về hay không
    if cropped_img is not None:
        # Hiển thị ảnh đã được cắt
        cv2.imshow("Cropped License Plate", cropped_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Không có biển số xe được phát hiện trong ảnh.")

# if __name__ == "__main__":
#     image_path = 'E:/PycharmProjects/License-Plate-Recognition-YOLOv7-and-CNN/xemay898.jpg'
#
#     # Gọi hàm return_crop_img để nhận ảnh đã được cắt từ biển số
#     cropped_img = return_crop_img(image_path)
#
#     # Kiểm tra xem ảnh đã được trả về hay không
#     if cropped_img is not None:
#         # Hiển thị ảnh đã được cắt
#         cv2.imshow("Cropped License Plate", cropped_img)
#
#         # Thêm code để hiển thị ảnh đã xoay ra
#         angle, rotate_thresh, LP_rotated = crop_n_rotate_LP(cropped_img, 0, 0, cropped_img.shape[1],
#                                                             cropped_img.shape[0])
#         if angle is not None and rotate_thresh is not None and LP_rotated is not None:
#             cv2.imshow("Rotated License Plate", LP_rotated)
#
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#     else:
#         print("Không có biển số xe được phát hiện trong ảnh.")

