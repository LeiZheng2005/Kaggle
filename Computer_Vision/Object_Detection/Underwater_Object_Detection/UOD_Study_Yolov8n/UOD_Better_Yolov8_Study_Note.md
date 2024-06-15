# 0.Better_Yolov8

这个代码的精度是高于上一个的，然后对比一下有哪些地方改进了，然后还有可以收获哪些内容。

>数据集：https://www.kaggle.com/datasets/slavkoprytula/aquarium-data-cots
>
>代码：kaggle：https://www.kaggle.com/code/quydau/underwater-object-detection-with-yolo-v8
>
>github：https://github.com/LeiZheng2005/Kaggle/tree/main/Computer_Vision/Object_Detection/Underwater_Object_Detection/UOD_Study_Yolov8n/UOD_Better_Yolov8.py

# 0.关于wandb

之前做目标检测的时候就用过所以也知道，然后这里的使用也是如下即可：

```python
import wandb

wandb.login(key="your_api_key")

```

api密钥在wandb官网里面可以找到，比较长。

# 1.关于训练

这里训练的时间明显比第一次代码的时间变长了，一轮需要20min左右。然后对比一下代码部分哪些地方是不一样的。

关于这个代码的结构的话是：首先定义一下路径和标签文件，获取基本属性，然后就是训练和验证。

路径部分：

```python
# 定义七个路径，六个数据和yaml配置文件
train_images = '/Users/leizheng/Dataset/Underwater_Object_Detection/aquarium_pretrain/train/images'
train_labels = '/Users/leizheng/Dataset/Underwater_Object_Detection/aquarium_pretrain/train/labels'

val_images = '/Users/leizheng/Dataset/Underwater_Object_Detection/aquarium_pretrain/valid/images'
val_labels = '/Users/leizheng/Dataset/Underwater_Object_Detection/aquarium_pretrain/valid/labels'

test_images = '/Users/leizheng/Dataset/Underwater_Object_Detection/aquarium_pretrain/test/images'
test_labels = '/Users/leizheng/Dataset/Underwater_Object_Detection/aquarium_pretrain/test/labels'

# Define the path to the yaml data file
yaml_path = '/Users/leizheng/Dataset/Underwater_Object_Detection/aquarium_pretrain/data.yaml'

```

像上面的虽然正确但是非常不合适，需要优化，比如利用os的join函数链接比较优雅。

```python
# 定义七个路径，六个数据和yaml配置文件
first_path = '/Users/leizheng/Dataset/Underwater_Object_Detection/aquarium_pretrain'
train_images = os.path.join(first_path,'train/images') 
train_labels = os.path.join(first_path,'train/labels') 
val_images = os.path.join(first_path,'valid/images') 
val_labels = os.path.join(first_path,'valid/labels') 
test_images = os.path.join(first_path,'test/images')
test_labels = os.path.join(first_path,'test/labels')
# Define the path to the yaml data file
yaml_path = os.path.join(first_path,'data.yaml')
```

修改之后也易于后续修改以及在别人电脑上复现。

```python
image_path = os.path.join(train_images, os.listdir(train_images)[100])
image = cv2.imread(image_path)
height, width, channels = image.shape
```

这一块利用os模块和cv模块对图像进行读取识别

往上部分的代码不会改变模型的精度，没有影响，往下开始改变。

首先是模型使用官方的预训练权重，这个和第一个代码一致，然后是将gpu的内存清空，没啥用。

调整为训练模式的时候对于图片大小进行了设置，seed没有多大影响，batch一样，workers还不知道是哪个参数，需要查一下，修改名称也没有影响。

```python
model = YOLO('yolov8n.pt')

# free up GPU memory
torch.cuda.empty_cache()

# Training the model
results = model.train(
    data=yaml_path,
    epochs=2,
    imgsz=(height, width, channels),
    seed=7,
    batch=16,
    workers=4,
    name='yolov8n_custom')
```

