import torch

from PIL import Image
from matplotlib import cm
from jetcam.usb_camera import USBCamera

from IPython.display import display
from jetcam.utils import bgr8_to_jpeg

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

webcam = USBCamera(width=224, height=224, capture_width=640, capture_height=480, capture_device=0)

image = webcam.read()
# print(webcam.shape)
# print(webcam.value.shape)

webcam.running = True

def update_image(change):
    image = change['new']
    im = Image.fromarray(image)
    im.show()
    result = model(image)
    result.print()
    im.close()

webcam.observe(update_image, names='value')