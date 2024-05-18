from ultralytics import YOLO
import cv2

# Load the YOLOv8 model
model = YOLO('E:/PycharmProjects/PblTrain/runs/detect/train/weights/best.pt')

# Perform inference on an image
results = model('E:/PycharmProjects/PblTrain/xemay916.jpg')

# Load the original image
image = "xemay916.jpg"
img = cv2.imread(image)

# Extract bounding boxes
boxes = results[0].boxes.xyxy.tolist()

# Iterate through the bounding boxes
for i, box in enumerate(boxes):
    x1, y1, x2, y2 = box
    # Crop the object using the bounding box coordinates
    ultralytics_crop_object = img[int(y1):int(y2), int(x1):int(x2)]
    # Save the cropped object as an image
    cv2.imwrite('ultralytics_crop_' + str(i) + '.jpg', ultralytics_crop_object)