from ultralytics import YOLO
from PIL import Image

# Load a pretrained YOLO model (recommended for training)
model = YOLO('E:/PycharmProjects/PBL/runs/detect/train/weights/best.pt')

results = model('E:/PycharmProjects/PBL/xemay916.jpg')

# Show image
for r in results:
    print(r.boxes)
    im_array = r.plot()
    im = Image.fromarray(im_array[..., ::-1])
    im.show()
    im.save('res.jpg')
