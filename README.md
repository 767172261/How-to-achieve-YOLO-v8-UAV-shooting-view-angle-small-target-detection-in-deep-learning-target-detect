YOLO v8无人机拍摄视角小目标检测数据集![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/450df66168874814a101c2d214c7f88d.png)

（如何训练这——9000张数据集图片，包括标签txt文件）

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a2b3aa986a1e4972a9e866af953009cd.png)

类别包括:
names:
0: pedestrian
1: people
2: bicycle
3: car
4: van
5: truck
6: tricycle
7: awning-tricycle
8: bus
9: motor![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8d9a1db7236d422a8463a4554b296378.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3c4e0ba19e8f4826a72f098eeddd1718.png)
声明：文章代码仅供参考：

实现一个基于 YOLOv8 的无人机拍摄视角小目标检测系统。以下是详细的步骤：

1. **数据准备**：确保数据集格式正确。
2. **环境部署**：安装必要的库。
3. **模型训练**：使用 YOLOv8 训练目标检测模型。
4. **PyQt5 GUI 开发**：创建一个简单的 GUI 来加载和运行模型进行实时预测。

### 数据集结构

假设你的数据集已经准备好，并且是以 YOLO 格式存储的。以下是数据集的标准结构：

```
dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── test/
│   │   ├── image3.jpg
│   │   ├── image4.jpg
│   │   └── ...
│   └── valid/
│       ├── image5.jpg
│       ├── image6.jpg
│       └── ...
├── labels/
│   ├── train/
│   │   ├── image1.txt
│   │   ├── image2.txt
│   │   └── ...
│   ├── test/
│   │   ├── image3.txt
│   │   ├── image4.txt
│   │   └── ...
│   └── valid/
│       ├── image5.txt
│       ├── image6.txt
│       └── ...
└── dataset.yaml
```

`dataset.yaml` 内容如下：
```yaml
train: ./images/train
val: ./images/valid
test: ./images/test

nc: 10
names: ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
```

每个图像对应的标签文件是一个文本文件，每行表示一个边界框，格式为：
```
<class_id> <x_center> <y_center> <width> <height>
```

### 环境部署说明

首先，确保你已经安装了必要的库。以下是详细的环境部署步骤：

#### 安装依赖

```bash
# 创建虚拟环境（可选）
conda create -n drone_detection_env python=3.9
conda activate drone_detection_env

# 安装PyTorch
pip install torch==1.9 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu111

# 安装其他依赖
pip install opencv-python pyqt5 ultralytics scikit-learn pandas matplotlib seaborn onnxruntime
```

### 模型训练权重和指标可视化展示

我们将使用 YOLOv8 进行目标检测任务。

#### 训练 YOLOv8

```python
[<title="Training YOLOv8 for Drone Small Object Detection">]
from ultralytics import YOLO
import os
import shutil

# Define paths
dataset_path = 'path/to/dataset'
weights_path = 'runs/train/exp/weights/best.pt'

# Create dataset.yaml
yaml_content = f"""
train: {os.path.join(dataset_path, 'images/train')}
val: {os.path.join(dataset_path, 'images/valid')}
test: {os.path.join(dataset_path, 'images/test')}

nc: 10
names: ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
"""

with open(os.path.join(dataset_path, 'dataset.yaml'), 'w') as f:
    f.write(yaml_content)

# Train YOLOv8
model = YOLO('yolov8n.pt')  # Load a pretrained model (recommended for training)
results = model.train(data=os.path.join(dataset_path, 'dataset.yaml'), epochs=100, imgsz=640, save=True)

# Save the best weights
best_weights_path = os.path.join('runs', 'train', 'exp', 'weights', 'best.pt')
shutil.copy(best_weights_path, weights_path)
```

请将 `path/to/dataset` 替换为实际的数据集路径。

### PyQt5 GUI 开发

我们将使用 PyQt5 创建一个简单的 GUI 来加载和运行 YOLOv8 模型进行实时预测。

#### 主窗口代码 `main_window.py`

```python
[<title="PyQt5 Main Window for Drone Small Object Detection">]
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from ultralytics import YOLO

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Drone Small Object Detection")
        self.setGeometry(100, 100, 800, 600)

        self.model = YOLO('runs/train/exp/weights/best.pt')

        self.initUI()

    def initUI(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

        self.load_button = QPushButton("Load Image", self)
        self.load_button.clicked.connect(self.load_image)
        self.layout.addWidget(self.load_button)

        self.detect_button = QPushButton("Detect Objects", self)
        self.detect_button.clicked.connect(self.detect_objects)
        self.layout.addWidget(self.detect_button)

        self.central_widget.setLayout(self.layout)

    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "Images (*.png *.xpm *.jpg *.jpeg);;All Files (*)", options=options)
        if file_name:
            self.image_path = file_name
            self.display_image(file_name)

    def display_image(self, path):
        pixmap = QPixmap(path)
        scaled_pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
        self.image_label.setPixmap(scaled_pixmap)

    def detect_objects(self):
        if hasattr(self, 'image_path'):
            image = cv2.imread(self.image_path)
            results = self.model(image)

            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    r = box.xyxy[0].astype(int)
                    cls = int(box.cls[0])
                    conf = box.conf[0]

                    label = self.model.names[cls]
                    text = f'{label}: {conf:.2f}'
                    color = (0, 255, 0)  # Green color for bounding box

                    cv2.rectangle(image, (r[0], r[1]), (r[2], r[3]), color, 2)
                    cv2.putText(image, text, (r[0], r[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
            self.image_label.setPixmap(scaled_pixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
```

### 使用说明

1. **配置路径**：
   - 将 `path/to/dataset` 设置为存放数据集的目录路径。
   - 确保 `runs/train/exp/weights/best.pt` 是训练好的 YOLOv8 模型权重路径。

2. **运行脚本**：
   - 在终端中运行 `main_window.py`。
   - 点击“Load Image”按钮加载图像。
   - 点击“Detect Objects”按钮进行目标检测，并在界面上显示结果。

3. **注意事项**：
   - 确保所有必要的工具箱已安装，特别是 PyTorch 和 PyQt5。
   - 根据需要调整参数，如 `epochs` 和 `imgsz`。

### 示例

假设你的数据文件夹结构如下：
```
dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── test/
│   │   ├── image3.jpg
│   │   ├── image4.jpg
│   │   └── ...
│   └── valid/
│       ├── image5.jpg
│       ├── image6.jpg
│       └── ...
├── labels/
│   ├── train/
│   │   ├── image1.txt
│   │   ├── image2.txt
│   │   └── ...
│   ├── test/
│   │   ├── image3.txt
│   │   ├── image4.txt
│   │   └── ...
│   └── valid/
│       ├── image5.txt
│       ├── image6.txt
│       └── ...
└── dataset.yaml
```

并且每个 `.txt` 文件中都有正确的 YOLO 标签。运行 `main_window.py` 后，可以通过点击按钮来加载图像并进行目标检测。

### 总结

构建一个完整的基于 YOLOv8 和 PyQt5 的无人机拍摄视角小目标检测系统，包括数据集准备、环境部署、模型训练、指标可视化展示和 PyQt5 GUI 开发。以下是所有相关的代码文件：

1. **训练 YOLOv8 脚本** (`train_yolov8.py`)
2. **PyQt5 主窗口代码** (`main_window.py`)

希望这些信息对你有所帮助---
