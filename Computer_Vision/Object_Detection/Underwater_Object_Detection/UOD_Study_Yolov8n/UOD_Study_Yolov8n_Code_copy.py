# !pip install ultralytics
# 这里对应mac就是使用终端环境pip3外部下载ultralytics
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO

root_dir = '/Users/leizheng/Dataset/Underwater_Object_Detection/aquarium_pretrain/data.yaml'
last_best_weight = '/Users/leizheng/PyCharm_Study_Code/DeepLearning_StudyCode/UnderWater_Object_Detection/UOD_Study_Yolov8n/runs/detect/train5/weights/best.pt '
model = YOLO(last_best_weight)
# model.train(
#     data = os.path.join(root_dir),
#     epochs = 1,
#     lr0 =0.0001
# )

# result = model.val()
result = '/Users/leizheng/Dataset/Underwater_Object_Detection/aquarium_pretrain/test/images/IMG_2289_jpeg_jpg.rf.fe2a7a149e7b11f2313f5a7b30386e85.jpg'
output = model(result)
from PIL import Image
for i, r in enumerate(output):
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image
    plt.axis('off')
    plt.imshow(im_rgb)
    plt.show()
