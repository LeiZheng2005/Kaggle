# 0.UOD_Yolov8n_Study

这是关于水下目标检测的数据集的一个代码，使用的是yolov8n，实验的代码精度不高可以说是非常低，但是没有关系，这次实验就是了解yolo是如何运用的就ok，一个简单的应用。下面会简单写一下这个代码的过程。

> 数据集地址：https://www.kaggle.com/datasets/slavkoprytula/aquarium-data-cots
>
> 代码地址：https://github.com/LeiZheng2005/Kaggle/tree/main/Computer_Vision/Object_Detection/Underwater_Object_Detection/UOD_Study_Yolov8n/UOD_Study_Yolov8n_Code_copy.py

下面是本次实验的收获：

# 1.关于环境配置

在mac中首先下载anaconda然后创建环境，这里需要注意的是，在基础环境之上使用pip3或者conda安装的时候提前看一下依赖包的版本，避免加载出错这个很重要。

以此代码为例：

```python
!pip install ultralytics
import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from IPython.display import Image, display
```

这些是我们需要的包，第一句是在kaggle上运行的所以这么下载没有问题，在pycharm中肯定不行的，需要用终端下载依赖包。

我们可以先这么查看版本：

```python
import ultralytics
import cv2
import matplotlib

print("Ultralytics 版本:", ultralytics.__version__)
print("OpenCV 版本:", cv2.__version__)
print("Matplotlib 版本:", matplotlib.__version__)
# Ultralytics 版本: 8.2.32
# OpenCV 版本: 4.9.0
# Matplotlib 版本: 3.7.5
```

然后使用conda指令创建环境之后，移动到指定环境中，然后使用pip3下载即可

```shell
(base) leizheng@LeiZhengsmini ~ % conda activiate UOD_Study_Yolov8n_Env
(UOD_Study_Yolov8n_Env) leizheng@LeiZhengsmini ~ % pip3 install ultralytics==8.2.32
```

像上面这样下载即可。

# 2.关于yolo加载

其实使用yolo挺简单的，按下面的步骤：

```python
# 首先是熟悉的导入包环节：
from ultralytics import YOLO

# 然后是选择yolo模型，可以使用如下指令，这里可以使用预训练的权重也可以使用yaml配置文件
model = YOLO(weight_path_or_yaml)
# 然后是调整为训练模式，设置适当的参数，其中date需要是yaml文件，具体格式参照下面的
# 训练之后可以设置为val验证模式，使用测试集的数据进行验证，
# 全部运行完之后会得到相关的图标和权重的
result = model.val()
```

对于结果可以进行单独的可视化操作，这里使用的是PIL中的Iamge函数进行。

```python
output = model(result)
from PIL import Image
for i, r in enumerate(output):
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image
    plt.axis('off')
    plt.imshow(im_rgb)
    plt.show()
```

对于上面提到的data中的yaml文件配置如下所示：

 ```yaml
 train: ../train/images
 val: ../valid/images
 test: ../test/images
 
 nc: 7
 names: ['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray']
 ```

# 3.关于下一篇

这个代码不是有问题嘛，就是精度只有0.0001好像，但是不重要，这次实验的意义在于明白了环境如何配置，如何更加正确的规范环境以及如何使用yolo模型，下一篇链接放这里，刚在kaggle上找到的，然后还有参考的文章也想一起看了。

>下一篇水下目标检测：
>
>数据集：同上：https://www.kaggle.com/datasets/slavkoprytula/aquarium-data-cots
>
>代码：kaggle：https://www.kaggle.com/code/quydau/underwater-object-detection-with-yolo-v8
>
>