[TOC]

图像就是矩阵，对图像操作，就是对矩阵操作

OpenCV 图像处理

- 图像基础知识
- 图像处理
- 图像分割
- 图像特征与目标检测
- 运动目标识别

OpenCV 学习目标  

- 了解图像处理基础知识，为学习 CV 方向论文打基础
- 通过系统学习 OpenCV 和图像处理知识，掌握常用的图像处理算法及代码实现
- 通过学习算法，可以将图像处理应用到论文撰写，求职面试等场景

---

# 一、图像基础知识

## 1.1 介绍

学习目标

- 数字图像为何物
- 数字图像的前世今生
- 链接图像各属性的含义
- 掌握常见的色彩空间

---

数字图像

- 数字图像，又称数码图像，一幅二维图像可以由一个数组或矩阵表示。
- 数字图像可以理解为一个二维函数 $f(x,y)$，其中 $x$ 和 $y$ 是空间(平面)坐标，而在任意坐标处的值 $f$ 称为图像在该点处的强度或灰度。

图像处理目的

- 改善图示的信息以便人们解释
- 为存储、传输和表示而对图像进行的处理

![Attachment.jpeg](./blob:file:/45af9979-63be-419e-9d3d-16db0d74f174) 

---

常见成像方式

- 电磁波谱
- $\gamma$ 射线成像
- $X$ 射线成像
- 紫外线波段成像
- 可见光波段成像
- 红外线波段成像
- 微波波段成像
- 射频波段成像

数字图像的应用

- 传统领域
  - 医学、空间应用、生物学、军事
- 最新领域
  - 数码相机(DC)、数码摄像机(DV)
  - 指纹识别、人脸识别，虹膜识别
  - 互联网、视频、多媒体等
  - 基于内容的图像检索、视频检索、多媒体检索
  - 水印、游戏、电影特技、虚拟现实、电子商务等  

---

图像处理 、 机器视觉 、 人工智能关系

- 图像处理主要研究==二维图像==，处理一个图像或一组图像之间的相互转换的过程，包括**图像滤波**，**图像识别**，**图像分割**等问题
- 计算机视觉主要研究映射到单幅或多幅图像上的==三维场景==，从图像中提取抽象的语义信息，实现图像理解是计算机视觉的终极目标
- 人工智能在计算机视觉上的目标就是==解决像素值和语义之间关系==，主要的问题有图片检测，图片识别，图片分割和图片检索

---

## 1.2 图像属性

### 1.2.1 图像格式

BMP格式

- Windows系统下的标准位图格式，未经过压缩，一般图像文件会比较大。在很多软件中被广泛应用.

JPEG格式

- 也是应用最广泛的图片格式之一，它采用一种特殊的有损压缩算法，达到较大的压缩比(可达到2:1甚至40:1)，互联网上最广泛使用的格式

GIF格式

- 不仅可以是一张静止的图片，也可以是动画，并且支持透明背景图像，适用于多种操作系统， “体型” 很小，网上很多小动画都是GIF格式。但是其色域不太广，只支持256种颜色.

PNG格式

- 与JPG格式类似，压缩比高于GIF，支持图像透明， 支持Alpha通道调节图像的透明度,

TIFF格式

- 它的特点是图像格式复杂、存贮信息多,在Mac中广泛使用，非常有利于原稿的复制。很多地方将TIFF格式用于印刷.  

### 1.2.2 图像尺寸

图像尺寸

- 图像尺寸的长度与宽度是以像素为单位的。

像素(pixel)

- 像素是数码影像最基本的单位，每个像素就是一个小点，而不同颜色的点聚集起来就变成一幅动人的照片。
- 灰度像素点数值范围在 0 到 255 之间， 0 表示黑、255 表示白，其它值表示处于黑白之间；
- 彩色图用红、绿、蓝三通道的二维矩阵来表示。每个数值也是在 0 到255 之间， 0 表示相应的基色，而 255 则代表相应的基色在该像素中取得最大值。

#### 1.2.2.1 读入图像  

函数： `cv2.imread()`

参数说明：

- 第一参数为待读路径
- 第二个参数为读取方式，常见读取方式有三种

| 读取方式                | 含义                                   | 数字表示 |
| ----------------------- | -------------------------------------- | -------- |
| `cv2.IMREAD_COLOR`      | 默认值，加载一张彩色图片，忽视透明度。 | 1        |
| `cv2.IMREAD_GRAYSC ALE` | 加载一张灰度图。                       | 0        |
| `cv2.IMREAD_UNCHAN GED` | 加载图像，包括它的 Alpha 通道。        | -1       |

```python
# 使用OpenCV中imread函数读取图片，
# 0 代表灰度图形式打开，1 代表彩色形式打开
img = cv2.imread('messi5.jpg', 0)
print(img.shape)
```

#### 1.2.2.2 显示图像

函数： `cv2.imshow()`

参数说明

- 参数1：窗口的名字
- 参数2：图像数据名

```python
# 调用imshow()西数进行图像显示
cv2.imshow('image'，img) # 窗口名字，数组变量名

# cv2.waithey() 是一个健盘绑定函数
# 单位毫秒，O代表等待健盘输入
cv2.waitKey(0) # 等待时间 0
# cv2.destroyAllWindows() 删除窗口，
# 默认值为所有窗口，参数一为待删除窗口名。
cv2.destroyAllWindows()
```

#### 1.2.2.3 保存图像

函数：` cv2.imwrite()`

参数说明

- 参数1：图像名（包括格式）
- 参数2：待写入的图像数据变量名。  

```python
cv2.imwrite('messigray.png', img) # 路径+名称，变量名
```

### 1.2.3 图像分辨率和通道数

分辨率

- 单位长度中所表达或截取的像素数目。每英寸图像内的像素点数，单位是像素每英寸(PPI)。图像分辨率越高，像素的点密度越高，图像越清晰。

通道数

- 图像的位深度，是指描述图像中每个pixel数值所占的二进制位数。 位深度越大则图像能表示的颜色数就越多，色彩越丰富逼真。
  - 8 位：单通道图像，也就是灰度图，灰度值范围 $2^8=256$
  - 24 位：三通道 $3\times8=24$
  - 32 位：三通道加透明度Alpha通道

#### 1.2.3.1 灰度转化

目的

- 将三通道图像（彩色图）转化为单通道图像（灰度图）
- 公式
  - $3 \to 1: \text{GRAY} = B \times 0.114 + G \times 0.587 + R \times 0.299$
  - $1 \to 3: R = G = B = \text{GRAY}; A = 0$
- 函数：`cv2.cvtColor(img, flag)`
- 参数说明
  - 参数 1：待转化图像
  - 参数 2： flag 就是转换模式， `cv2.COLOR_BGR2GRAY`：彩色转灰度
  - `cv2.COLOR_GRAY2BGR`：单通道转三通道

```python
#导入opencv
import cv2

#读入原始图像，使用cv2.IMREAD_UNCHANGED
img = cv2.imread("girl.jpg", cv2.IMREAD_UNCHANGED)
#查看打印图像的shape
shape = img.shape
print(shape)
#判断通道数是否为3通道或4通道
if shape[2] == 3 or shape[2] == 4:
    #将彩色图转化为单通道图
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 图像变量 转化模式
    cv2.imshow("gray_image", img_gray)
cv2.imshow("image", img)
cv2.waitKey(1000)
cv2.destroyAllWindows()
```

#### 1.2.3.2 RGB与BGR转化

<img src="https://s2.loli.net/2023/10/25/NBELVX4AFPTC5n8.png" alt="img" style="zoom:80%;" />

```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("test2.png", cv2.IMREAD_COLOR)
cv2.imshow("Opencv_win", img)
# 用opencv自带的方法转
img_cv_method = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 用numpy转，img[:,:,::-1]列左右翻转
img_numpy_method = img[:, :, ::-1]  # 本来是BGR 现在逆序，变成RGB
# 用matplot画图
plt.subplot(1, 3, 1)
plt.imshow(img_cv_method)
plt.subplot(1, 3, 2)
plt.imshow(img_numpy_method)
plt.subplot(1, 3, 3)
plt.imshow(img) # 原图
plt.savefig("./plt.png")
plt.show()
#保存图片
cv2.imwrite("opencv.png", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

#### 1.2.3.3 通道分离

目的

- 将彩色图像，分成b、 g、 r 3个单通道图像。方便我们对 BGR 三个通道分别进行操作。

函数：`cv2.split(img)`

参数说明

- 参数 1：待分离通道的图像

```python
#加载opencv
import cv2

src = cv2.imread('split.jpg')
cv2.imshow('before', src)
#调用通道分离
b, g, r = cv2.split(src)
#三通道分别显示，单通道一定是灰度图
cv2.imshow('blue', b)
cv2.imshow('green', g)
cv2.imshow('red', r)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 如果想要把三个通道的颜色展现出来
zeros = np.zeros(image.shape[:2], dtype="uint8")  #创建与image相同大小的零矩阵
cv2.imshow("BLUE", cv2.merge([b, zeros, zeros]))  #显示 （B，0，0）图像
cv2.imshow("GREEN", cv2.merge([zeros, g, zeros]))  #显示（0，G，0）图像
cv2.imshow("RED", cv2.merge([zeros, zeros, r]))  #显示（0，0，R）图像
cv2.waitKey(0)
cv2.destroyAllWindows()
```

#### 1.2.3.4 通道合并

目的

- 通道分离为 B,G,R 后，对单独通道进行修改，最后将修改后的三通道合并为彩色图像。

函数： `cv2.merge(List)`

参数说明

- 参数1：待合并的通道数，以 list 的形式输入  

```python
#加载opencv
import cv2

src = cv2.imread('split.jpg')
cv2.imshow('before', src)
#调用通道分离
b, g, r = cv2.split(src)
#将Blue通道数值修改为0
g[:] = 0
#合并修改后的通道
img_merge = cv2.merge([b, g, r])
cv2.imshow('merge', img_merge)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 1.2.4 图像直方图  

图像直方图

- 图像直方图（Image Histogram）是用以表示数字图像中亮度分布的直方图，标绘了图像中每个亮度值的像素数。这种直方图中，横坐标的左侧为纯黑、较暗的区域，而右侧为较亮、纯白的区域。

<center class="half">
    <img src="https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220925092700062.png" width="300"/>
    <img src="https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220925092707542.png" width="300"/>
</center>

图像直方图的意义

- 直方图是图像中像素强度分布的图形表达方式。
- 它统计了每一个强度值所具有的像素个数
- CV 领域常借助图像直方图来实现图像的二值化

#### 1.2.4.1 直方图绘制

目的

- 直方图是对图像像素的统计分布，它统计了每个像素（0到255）的数量。

函数

- `cv2.calcHist(images, channels, mask, histSize, ranges)`

参数说明

- 参数1：待统计图像，需用**中括号**括起来
- 参数2：待计算的通道（0,1,2->b,g,r）
- 参数3：`Mask`，这里没有使用，所以用 `None`。
- 参数4：`histSize`，表示直方图分成多少份；
- 参数5：是表示直方图中各个像素的值， `[0.0, 256.0]` 表示直方图能表示像素值从 0.0到 256 的像素。直方图是对图像像素的统计分布，它统计了每个像素（0到255）的数量。

```python
from matplotlib import pyplot as plt
import cv2
import numpy as np

img = cv2.imread('girl.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.imshow(img_gray, cmap=plt.cm.gray)
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
# 图像，通道数，mask，份数，灰度值范围

plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist)
plt.xlim([0, 256])
plt.show()
```

#### 1.2.4.2 三通道直方图绘制

```python
from matplotlib import pyplot as plt
import cv2

girl = cv2.imread("girl.jpg")
cv2.imshow("girl", girl)
color = ("b", "g", "r")
#使用for循环遍历color列表，enumerate枚举返回索引和值
for i, color in enumerate(color):
    hist = cv2.calcHist([girl], [i], None, [256], [0, 256])
    plt.title("girl")
    plt.xlabel("Bins")
    plt.ylabel("num of perlex")
    plt.plot(hist, color=color)
    plt.xlim([0, 260])
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 1.2.5 颜色空间

概念

- 颜色空间也称彩色模型(又称彩色空间或彩色系统）它的用途是在某些标准下用通常可接受的方式对彩色加以说明。

常见的颜色空间

- RGB、 HSV、 HSI、 CMYK

---

#### 1.2.5.1 RGB

概念

- 主要用于计算机图形学中，依据人眼识别的颜色创建，图像中每一个像素都具有R,G,B三个颜色分量组成，这三个分量大小均为[0,255]。通常表示某个颜色的时候，写成一个3维向量的形式（110,150,130）。

<img src="https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220925002342609.png" alt="image-20220925002342609" style="zoom:80%;" />

颜色模型

- 原点对应的颜色为黑色，它的三个分量值都为0；
- 距离原点最远的顶点对应的颜色为白色，三个分量值都为1；
- 从黑色到白色的灰度值分布在这两个点的连线上，该虚线称为灰度线；
- 立方体的其余各点对应不同的颜色，即三原色红、绿、蓝及其混合色黄、品红、 青色；

---

#### 1.2.5.2 HSV

概念

- HSV(Hue, Saturation, Value)是根据颜色的直观特性由A. R. Smith在1978年创建的一种颜色空间,这个模型中颜色的参数分别是：色调（H），饱和度（S），明度（V）。

颜色模型

- H通道： Hue，色调/色彩，这个通道代表颜色。
- S通道： Saturation，饱和度，取值范围 0%～ 100%，值越大，颜色越饱和。
- V通道： Value，明暗，数值越高，越明亮， 0%（黑）到100%（白）。  

<img src="https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220925093440050.png" alt="image-20220925093440050" style="zoom: 43%;" />

#### 1.2.5.3 RGB空间与HSV转化

```python
import cv2


#色彩空间转换函数
def color_space_demo(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', gray)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #print(hsv)
    cv2.imshow('hsv', hsv)


#读入一张彩色图
src = cv2.imread('girl.jpg')
cv2.imshow('before', src)
#调用color_space_demo函数进行色彩空间转化
color_space_demo(src)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

![image-20220925093835015](https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220925093835015.png)

---

#### 1.2.5.4 HSI

概念

- HSI模型是美国色彩学家孟塞尔(H.A.Munseu)于1915年提出的，它反映了人的视觉系统感知彩色的方式，以色调、饱和度和强度三种基本特征量来感知颜色。

模型优点

- 在处理彩色图像时，可仅对I分量进行处理，结果不改变原图像中的彩色种类；
- HSI模型完全反映了人感知颜色的基本属性，与人感知颜色的结果一一对应。  

![image-20220925093610582](https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220925093610582.png)

---

#### 1.2.5.5 CMYK

概念

- CMYK(Cyan, Magenta,Yellow, black)颜色空间应用于印刷工业,印刷业通过青(C)、品(M)、黄(Y)三原色油墨的不同网点面积率的叠印来表现丰富多彩的颜色和阶调，这便是三原色的CMY颜色空间。  

![image-20220925093632539](https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220925093632539.png)

## 1.3 空间域、频域

空间域：计算简单，可以实时在图片上反映出来

频域：计算复杂，可以实时反映出来，但是高效

# 二、图像基本操作

## 2.1 绘图函数

### 2.1.1 线段绘制

函数： `cv2.line(img, pts, color, thickness, linetype)`

参数说明
- `img`：待绘制图像（底板，画板）
- `color`：形状的颜色，元组如 `(255,0,0)`
- `pts`：起点和终点
- `thickness`：线条的粗细。`-1` 为填充，默认值是 `1`
- `linetype`：线条的类型，`8` 型或 `cv2.LINE_AA`，默认值为 `8` 型。  

```python
# 创建一张黑色的背景图
img=np.zeros((512,512,3), np.uint8)

# 绘制一条线宽为5的线段
cv2.line(img,(0,0),(511,511),(255,0,0),5)
# 背景图，起点，终点，线段颜色，线宽
```

### 2.1.2 矩形绘制

函数： `cv2.rectangle(img, prets, color, thickness, linetype)`

参数说明：
- `img`：待绘制图像
- `pts`：左上角和右下角坐标点
- `color`：形状的颜色，元组如（255,0,0）
- `thickness`：线条的粗细。 -1 为填充，默认值是 1
- `linetype`：线条的类型，使用默认值即可

```python
# 创建一张黑色的背景图
img=np.zeros((512,512,3), np.uint8)

# 画一个绿色边框的矩形，参数2：左上角坐标，参数3：右下角坐标
cv2.rectangle(img,(384,0),(510,128),(0,255,255),1)
```

### 2.1.3 圆绘制

函数：`cv2.circle(img, pts, radius, color, thickness, linetype)`

参数说明： 

- `img`：待绘制图像
- `pts`：圆心
- `radius`：半径
- `color`：颜色
- `thickness`：线条的粗细。`-1` 为填充，默认值是 `1`
- `linetype`：线条的类型，使用默认值即可

```python
# 创建一张黑色的背景图
img=np.zeros((512,512,3), np.uint8)

# 画一个填充红色的圆，参数2：圆心坐标，参数3：半径
cv2.circle(img,(447,63), 63, (0,0,255), -1)
# 圆心，半径，颜色，填充
```

### 2.1.4 椭圆绘制

函数：`cv2.ellipse()`

画椭圆需要的参数比较多，请对照后面的代码理解这几个参数：

参数说明：

- 参数2：椭圆中心 `(x,y)`
- 参数3：`x/y` 轴的长度
- 参数4：`angle` 椭圆的旋转角度
- 参数5：`startAngle` 椭圆的起始角度
- 参数6：`endAngle` 椭圆的结束角度

```python
# 创建一张黑色的背景图
img=np.zeros((512,512,3), np.uint8)

# 在图中心画一个填充的半圆
cv2.ellipse(img, (256, 256), (100, 50), 0, 30, 180, (255, 0, 0), -1)
# 中心点坐标，(长，短轴)，旋转角，起始，终止
```

### 2.1.5 多边形绘制

函数：`cv2.polylines(img, pts, isClosed, color, thickness, lineType)`

参数说明：

- 参数1：`img` 图像，表示你要在哪张图像上画线
- 参数2：`pts`，表示的是点对，形式如下：
- 参数3：`isClosed`，布尔型，True 表示的是线段闭合，False 表示的是仅保留线段
- 参数4：`color`，线段颜色，格式是（ R,G,B）值
- 参数5：`thickness`, 数值型，线宽度，默认值为1， -1则会填充整个图形
- 参数6：`lineType`，线型

```python
# 创建一张黑色的背景图
img=np.zeros((512,512,3), np.uint8)

# 定义四个顶点坐标
pts = np.array([[10, 5],  [50, 10], [70, 20], [20, 30]])
print(pts)
# 顶点个数：4，矩阵变成4*1*2维
pts = pts.reshape((-1, 1, 2))
print(pts)
#绘制椭圆
cv2.polylines(img, [pts], False, (0, 255, 255))
winname = 'example'
cv2.namedWindow(winname)
cv2.imshow(winname, img)
cv2.waitKey(0)
cv2.destroyWindow(winname)
```

### 2.1.6 添加文字

函数：`cv2.putText()`

同样请对照后面的代码理解这几个参数：

- 参数2：要添加的文本
- 参数3：文字的起始坐标（左下角为起点）
- 参数4：字体
- 参数5：文字大小（缩放比例）
- 参数6：颜色
- 参数7：线条宽度
- 参数8：线条形状

```python
import numpy as np
import cv2

# 创建一张黑色的背景图
img=np.zeros((512,512,3), np.uint8)

#添加文字
font=cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'OpenCV',(50,200), font, 3,(0,255,255),5, cv2.LINE_AA)
# 待写文字，起点，字体，缩放比例，颜色，线宽，线条形状(默认 8 型)
```

### 2.1.7 综合效果

<img src="https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220925100757035.png" alt="image-20220925100757035" style="zoom:67%;" />

## 2.2 图像的几何变换

### 2.2.1 图像平移

- 将图像中所有的点按照指定的平移量水平或者垂直移动

变换公式

- 设 $(x_0,y_0)$ 为原图像上的一点，图像水平平移量为 $T_x$，垂直平移量为 $T_y$，则平移后的点坐标 $(x_1,y_1)$ 变为

$$
x_1=x_0+T_x\\
y_1=y_0+T_y
$$

---

仿射变换函数： `cv2.warpAffine(src, M, dsize，borderMode, borderValue)`

其中：

- `src` - 输入图像
- `M` - 变换矩阵 $2\times3$
- `dsize` - 输出图像的大小。
- `flags` - 插值方法的组合（ `int` 类型！）
- `borderMode` - 边界像素模式（ `int` 类型！）
- `borderValue` - （重点！）边界填充值; 默认情况下为 0

上述参数中

- M作为仿射变换矩阵，一般反映平移或旋转的关系，为 `InputArray` 类型的 2×3 的变换矩阵。
- `flages` 表示插值方式，默认为 `flags=cv2.INTER_LINEAR`，表示线性插值，

此外还有

- `cv2.INTER_NEAREST`（最近邻插值）
- `cv2.INTER_AREA` （区域插值）
- `cv2.INTER_CUBIC`（三次样条插值）
- `cv2.INTER_LANCZOS4`（ Lanczos 插值）  

<img src="https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220925112238355.png" alt="image-20220925112238355" style="zoom:80%;" />

```py
import cv2
import numpy as np
img = cv2.imread('img2.png')
# 构造移动矩阵H
# 在x轴方向移动多少距离，在y轴方向移动多少距离
H = np.float32([[1, 0, 50], [0, 1, 25]])
rows, cols = img.shape[:2]
print(img.shape)
print(rows, cols)

# 注意这里rows和cols需要反置，即先列后行
res = cv2.warpAffine(img, H, (2*cols, 2*rows))  
cv2.imshow('origin_picture', img)
cv2.imshow('new_picture', res)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 2.2.2 图像缩放

#### 2.2.2.1 上采样和下采样

下采样：缩小图像称为下采样（subsampled）或降采样（downsampled）

上采样：放大图像称为上采样（upsampling），主要目的得到更高分辨率图像。

![image-20220925121150844](https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220925121150844.png)

图像缩放： 图像缩放是指图像大小按照指定的比例进行放大或者缩小。

函数：`cv2.resize(src,dsize=None,fx,fy,interpolation)`

- `src`：原图
- `dsize`：输出图像尺寸，与比例因子二选一
- `fx`：沿水平轴的比例因子
- `fy`：沿垂直轴的比例因子
- `interpolation`：插值方法  

#### 2.2.2.2 插值法

- 默认为 `flags=cv2.INTER_NEAREST`（最近邻插值）
- `cv2.INTER_LINEAR` 线性插值
- `cv2.INTER_CUBIC` 三次样条插值 $4\times4$ 像素邻域
- `cv2.INTER_LANCZOS4` Lanczos 插值， $8\times8$ 像素邻域
- `cv2.INTER_AREA` 区域插值  

---

##### 最近邻插值

最简单的一种插值方法，不需要计算，在待求像素的四邻像素中，将距离待求像素最近的邻像素灰度赋给待求像素。

设 $i+u, j+v$（$i,j$ 为正整数， $u, v$ 为大于零小于 1 的小数，下同）为待求像素坐标，则待求像素灰度的值 $f(i+u, j+v)$

公式如下

- $\rm srcX=dstX \times (srcWidth/dstWidth)$
- $\rm srcY = dstY \times (srcHeight/dstHeight)$

![image-20220925123457524](https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220925123457524.png)
$$
\begin{aligned}
(0\times(3/4),0\times(3/4))&\Rightarrow(0\times0.75,0\times0.75)&\Rightarrow(0,0)&\Rightarrow234\\

\left(1\times0.75,0\times0.75\right)&\Rightarrow(0.75,0)&\Rightarrow(1,0)&\Rightarrow 67\\

\left(2\times0.75,0\times0.75\right)&\Rightarrow(1.5,0)&\Rightarrow(2,0)&\Rightarrow 89\\

\left(3\times0.75,0\times0.75\right)&\Rightarrow(2.25,0)&\Rightarrow(2,0)&\Rightarrow 89
\end{aligned}
$$

##### 双线性插值

双线性插值又叫一阶插值法，它要经过三次插值才能获得最终结果，是对最近邻插值法的一种改进，先对两水平方向进行一阶线性插值，然后再在垂直方向上进行一阶线性插值。  

<img src="https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220925122312266.png" alt="image-20220925122312266" style="zoom:50%;" />
$$
\begin{align}
\frac{y-y_0}{x-x_0}&=\frac{y_1-y_0}{x_1-x_0}\\
y=\frac{x_1-x}{x_1-x_0}&+\frac{x-x_0}{x_1-x_0}
\end{align}
$$

<img src="https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220925122334787.png" alt="image-20220925122334787" style="zoom:50%;" />
$$
\begin{align}
f\left(R_1\right) &\approx \frac{x_2-x}{x_2-x_1} f\left(Q_{11}\right)+\frac{x-x_1}{x_2-x_1} f\left(Q_{21}\right) \text { where } R_1=\left(x, y_1\right) \\

f\left(R_2\right) &\approx \frac{x_2-x}{x_2-x_1} f\left(Q_{12}\right)+\frac{x-x_1}{x_2-x_1} f\left(Q_{22}\right) \text { where } R_2=\left(x, y_2\right) \\

f(P) &\approx \frac{y_2-y}{y_2-y_1} f\left(R_1\right)+\frac{y-y_1}{y_2-y_1} f\left(R_2\right) . \\

f(x, y) &\approx \frac{f\left(Q_{11}\right)}{\left(x_2-x_1\right)\left(y_2-y_1\right)}\left(x_2-x\right)\left(y_2-y\right)+\frac{f\left(Q_{21}\right)}{\left(x_2-x_1\right)\left(y_2-y_1\right)}\left(x+x_1\right)\left(y_2-y\right) \\

&+\frac{f\left(Q_{12}\right)}{\left(x_2-x_1\right)\left(y_2-y_1\right)}\left(x_2-x\right)\left(y-y_1\right)+\frac{f\left(Q_{22}\right)}{\left(x_2-x_1\right)\left(y_2-y_1\right)}\left(x-x_1\right)\left(y-y_1\right) .
\end{align}
$$

```py
img = cv2.imread('img2.png')
# 方法一：通过设置缩放比例，来对图像进行放大或缩小
res1 = cv2.resize(img, None, fx=2, fy=2, 
                  interpolation=cv2.INTER_CUBIC)
height, width = img.shape[:2]

# 方法二：直接设置图像的大小，不需要缩放因子
res2 = cv2.resize(img, (int(0.8*width),
                        int(0.8*height)),interpolation=cv2.INTER_LANCZOS4)
cv2.imshow('origin_picture', img)
cv2.imshow('res1', res1)
cv2.imshow('res2', res2)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 2.2.3 图像旋转

以图像的中心为原点，旋转一定的角度，也就是将图像上的所有像素都旋转一个相同的角度。旋转后图像的的大小一般会改变，即可以把转出显示区域的图像截去，或者扩大图像范围来显示所有的图像。图像的旋转变换也可以用矩阵变换来表示。  

设点 $P_0(x_0,y_0)$ 逆时针旋转 $\theta$ 角后的对应点为 $P(x,y)$，那么，旋转后点 $P(x,y)$ 的坐标是  
$$
\left\{\begin{array}{l}
x=r \cos (\alpha+\theta)=r \cos \alpha \cos \theta-r \sin \alpha \sin \theta=x_0 \cos \theta-y_0 \sin \theta \\
y=r \sin (\alpha+\theta)=r \sin \alpha \cos \theta+r \cos \alpha \sin \theta=x_0 \sin \theta+y_0 \cos \theta
\end{array}\right.
$$
利用上述方法进行图像旋转时需要注意如下两点

1. 图像旋转之前，为了避免信息的丢失，一定要有坐标平移。
2. 图像旋转之后，会出现许多空洞点。对这些空洞点必须进行填充处理，否则画面效果不好，一般也称这种操作为插值处理。

---

变换矩阵函数：`cv2.getRotationMatrix2D(center, angle, scale)`

参数

- `center`：图片的旋转中心
- `angle`：旋转角度
- `scale`：缩放比例， 0.5表示缩小一半
- 正为逆时针，负值为顺时针  

```py
img=cv2.imread('img2.png',1)
rows,cols=img.shape[:2]
#参数1：旋转中心，参数2：旋转角度，参数3：缩放因子
#参数3 正为逆时针，负值为正时针
M=cv2.getRotationMatrix2D((cols/2,rows/2),45,1,)
print(M)
#第三个参数是输出图像的尺寸中心
dst=cv2.warpAffine(img,M,(cols,rows)) # 默认用 0 来填充
#dst=cv2.warpAffine(img,M,(cols,rows),borderValue=(255,255,255))
while(1):
    cv2.imshow('img', img)
    cv2.imshow('img1', dst)
    #0xFF==27  ESC
    if cv2.waitKey(1)&0xFF==27:
        break
cv2.destroyAllWindows()
```

### 2.2.4 仿射变换

仿射变换的作用
- 通过仿射变换对图片进行旋转、平移、缩放等操作以达到数据增强的效果

线性变换从几何直观上来看有两个要点
- 变换前是直线，变换后依然是直线
- 直线的比例保持不变

<img src="https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220925125918219.png" alt="image-20220925125918219" style="zoom: 50%;" />


仿射变换：平移、旋转、放缩、剪切、反射

仿射变换的函数原型如下

- `M = cv2.getAffineTransform(pos1,pos2)`
- `pos1` 表示变换前的位置
- `pos2` 表示变换后的位置  

```py
#读取图片
src = cv2.imread('bird.png')
#获取图像大小
rows, cols = src.shape[:2]
#设置图像仿射变换矩阵
pos1 = np.float32([[50,50], [200,50], [50,200]])
pos2 = np.float32([[10,100], [200,50], [100,250]])
M = cv2.getAffineTransform(pos1, pos2)
print(M)
#图像仿射变换
result = cv2.warpAffine(src, M, (2*cols, 2*rows))
#显示图像
cv2.imshow("original", src)

cv2.imshow("result", result)
#等待显示
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<center class="half">
    <img src="https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220925125932205.png" width="300"/>
    <img src="https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220925125935429.png" width="300"/>
</center>
### 2.2.5 透视变换

本质是将图像投影到一个新的视平面。

函数：

- `M = cv2.getPerspectiveTransform(pos1, pos2)`
  - `pos1` 表示透视变换前的 4 个点对应位置
  - `pos2` 表示透视变换后的 4 个点对应位置
- `cv2.warpPerspective(src,M,(cols,rows))`
  - `src` 表示原始图像
  - `M` 表示透视变换矩阵
  - `(rows,cols)` 表示变换后的图像大小， `rows` 表示行数，`cols` 表示列数  

```py
#读取图片
src = cv2.imread('bird.png')
#获取图像大小
rows, cols = src.shape[:2]
#设置图像透视变换矩阵
pos1 = np.float32([[114, 82], [287, 156],
                   [8, 100], [143, 177]])
pos2 = np.float32([[0, 0], [188, 0],
                   [0, 262], [188, 262]])
M = cv2.getPerspectiveTransform(pos1, pos2)
#图像透视变换
result = cv2.warpPerspective(src, M, (2*cols,2*rows))
#显示图像
cv2.imshow("original", src)
cv2.imshow("result", result)
#等待显示
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<center class="half">
    <img src="https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220925125932205.png" width="300"/>
    <img src="https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220925130233233.png" width="300"/>
</center>

## 2.3 图像滤波

### 2.3.1 图像滤波简介

滤波实际上是信号处理得一个概念，图像可以看成一个二维信号，其中像素点的灰度值代表信号的强弱；

高低频滤波

- 高频：图像上变化剧烈的部分；
- 低频：图像灰度值变化缓慢，平坦的地方；
- 根据图像高低频，设置高通和低通滤波器。高通滤波器可以检测变化尖锐，明显的地方，低通可以让图像变得平滑，消除噪声；
- 滤波作用：高通滤波器用于<u>**边缘检测**</u>，低通滤波器用于图像<u>**平滑去噪**</u>；

滤波器种类

- 线性滤波：方框滤波/均值滤波/高斯滤波
- 非线性滤波：中值滤波/双边滤波；

领域算子：利用给定像素周围的像素值，决定此像素的最终输出值的一种算子；

线性滤波：一种常用的领域算子，像素输出取决于输入像素的加权和：  
$$
g(i,j)=\sum\limits_{k,l}f(i+k,j+l)h(k,l)
$$
$h(k,l)$ 是 kernel，也就是 CNN 里面的卷积算子

### 2.3.2 线性滤波

#### a. 方框滤波

方框滤波（ box Filter）被封装在一个名为 `boxFilter` 的函数中，即 `boxFilter` 函数的作用是使用方框滤波器（ box filter）来模糊一张图片，从 `src` 输入，从 `dst` 输出；

方框滤波核：

$$
\begin{align}
{\rm{K}} &= \alpha \left[ {\begin{array}{*{20}{c}}
1&1& \cdots &1\\
1&1& \cdots &1\\
 \cdots & \cdots & \cdots & \cdots \\
1&1& \cdots &1
\end{array}} \right]\\

\alpha&=
\left\{ {\begin{array}{*{20}{cr}}
{\frac{1}{{{\rm{width}} \times {\rm{height}}}}}&{\quad {\rm{normalize}} = {\rm{true}}}\\
1&{{\rm{normalize}} = {\rm{false}}}
\end{array}} \right.
\end{align}
$$

- `normalize = true` 与均值滤波相同（归一化）
- `normalize = false` 很容易发生溢出（图片像素变白）

函数：`cv2.boxFilter(src, depth, ksize, normalize)`

参数说明

- 参数1：输入图像
- 参数2：目标图像深度
- 参数3：核大小
- 参数4：normalize 属性  

```py
img = cv2.imread('girl2.png',cv2.IMREAD_UNCHANGED)
r = cv2.boxFilter(img, -1 , (7,7) , normalize = 1)
d = cv2.boxFilter(img, -1 , (3,3) , normalize = 0)
cv2.namedWindow('img',cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('r',cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('d',cv2.WINDOW_AUTOSIZE)
cv2.imshow('img',img)
cv2.imshow('r',r)
cv2.imshow('d',d)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

#### b. 均值滤波

neighborhood average

均值滤波是一种最简单的滤波处理，它取的是卷积核区域内元素的均值，用 `cv2.blur()` 实现，如 $3\times3$ 的卷积核：
$$
{\rm{kernel}} = \frac{1}{9}\left[ {\begin{array}{*{20}{c}}
1&1&1\\
1&1&1\\
1&1&1
\end{array}} \right]
$$
smoothing in x and y

函数：`cv2.blur(src, ksize)`

参数说明

- 参数1：输入原图
- 参数2： kernel 的大小，一般为奇数

```py
img = cv2.imread('image/opencv.png')
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # 在plt中默认是RGB
blur = cv2.blur(img,(3,3))
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()
```

#### c. 高斯滤波

高斯滤波是一种线性平滑滤波，适用于消除高斯噪声，广泛应用于图像处理的减噪过程。高斯滤波的卷积核权重并不相同，中间像素点权重最高，越远离中心的像素权重越小。其原理是一个 2 维高斯函数）
$$
{\rm{kernel}} = \frac{1}{16}\left[ {\begin{array}{*{20}{c}}
1&2&1\\
2&4&2\\
1&2&1
\end{array}} \right]
$$

<center class="half">
    <img src="https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220925185859736.png" width="350"/>
    <img src="https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220925185920709.png" width="250"/>
</center>
高斯滤波相比均值滤波效率要慢，但可以有效消除高斯噪声，能保留更多的图像细节，所以经常被称为最有用的滤波器。

- 它是唯一可分离且圆对称的滤波器
- 在空间和频域具有最佳联合定位
- 高斯的傅里叶变换也是高斯函数
- 任何低通滤波器的 $n$ 倍卷积都收敛到高斯
- 它是无限平滑的，因此可以微分到任何所需的程度
- 它自然缩放（sigma）并允许一致的尺度空间理论
- 如果必须保留小对象，则最好使用高斯滤波

---

函数：`cv2.Guassianblur(src, ksize, std)`

参数说明

- 参数1：输入原图
- 参数2：高斯核大小
- 参数3：标准差 $\sigma$，平滑时，调整 $\sigma$ 实际是在调整周围像素对当前像素的影响程度，调大 $\sigma$ 即提高了远处像素对中心像素的影响程度，滤波结果也就越平滑。  

```py
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread('image/median.png')
img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
blur = cv.GaussianBlur(img,(7,7),7)
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()
```

### 2.3.3 非线性滤波

都会改变图像本身的性质，所以在医学图像中要慎用

#### a. 中值滤波

中值滤波是一种非线性滤波，是用像素点邻域灰度值的中指代替该点的灰度值，中值滤波可以去除椒盐噪声和斑点噪声。

- 强制具有不同强度的像素更像它们的邻居
- 它消除了孤立的强度尖峰（salt and pepper image noise）
- 邻域的大小通常为 $n\times n$ 像素，$n = 3, 5, 7$
- 这也消除了 pixel clusters（亮或暗）面积 $n^2/2$
- 如果必须删除小对象，中值过滤是最好的

函数：`cv2.medianBlur(img,ksize)`

参数说明:

- 参数1：输入原图
- 参数2：核大小  

```py
img = cv.imread('image/median.png')
median = cv.medianBlur(img,3)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(median),plt.title('median')
plt.xticks([]), plt.yticks([])
plt.show()
```

#### b. 双边滤波

双边滤波是一种非线性的滤波方法，是结合图像的空间邻近度和像素值相似度的一种折中处理，同时考虑空间与信息和灰度相似性，达到==保边去噪==的目的，具有简单、非迭代、局部处理的特点。

函数： `cv2.bilateralFilter(src=image, d,sigmaColor, sigmaSpace)`

参数说明

- 参数1：输入原图
- 参数2：像素的邻域直径
- 参数3： 灰度值相似性高斯函数标准差
- 参数4： 空间高斯函数标准差  

```py
img = cv2.imread('image/bilateral.png')
img = cv2.cvtColor(img,cv.COLOR_BGR2RGB)
blur = cv2.bilateralFilter(img,-1,15,10)
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()
```

## 2.4 图像增强

### 2.4.1 直方图均衡化

Histogram Procesing（重点）

- Histogram Equalization 均衡
  - 在整个亮度范围内获得具有均匀分布的亮度级别的图像
  - 在图像对比度低的时候可以使用
- Histogram Matching
  - 获取具有指定直方图（亮度分布）的图像

---

理论计算

- 对于离散值，我们得到 probabilities 和 summations，而不是 p.d.fs 和积分

$$
p_r(r_k)=n_k/(MN),\quad k=0,1,...,L-1
$$

- 其中 $MN$ 是图像中的总像素数，$n_k$ 是具有灰度值为 $r_k$ 的像素数，$L$ 是总灰度值数，$k$ 是 0 到 255 中的数

所以
$$
s_k=T(r_k)=(L-1)\sum\limits_{j=0}^k p_r(r_j)=\frac{L-1}{MN}\sum\limits_{j=0}^k n_j\\
k=0,1,...,L-1
$$
这种变换称为直方图均衡

然而，在实际应用中，获得完全均匀分布（离散版本）是很少见的。

和灰度值有关的变量有 2 个 ->  $k$ 和 $r_k$

---

<img src="https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220925214052204.png" alt="image-20220925214052204" style="zoom:80%;" />

目的

- 直方图均衡化是将原图像通过某种变换，得到一幅灰度直方图为均匀分布的新图像的方法。
- 直方图均衡化方法的基本思想是对在图像中像素个数多的灰度级进行展宽，而对像素个数少的灰度级进行缩减。从而达到清晰图像的目的。

函数：`cv2.equalizeHist(img)`

- 参数1： 待均衡化图像

步骤

- 统计直方图中每个灰度级出现的次数
- 计算累计归一化直方图
- 重新计算像素点的像素值

---

灰度直方图均衡化

```py
#直接读为灰度图像
img = cv2.imread('./image/dark.png',0)
cv2.imshow("dark",img)
cv2.waitKey(0)
#调用cv2.equalizeHist函数进行直方图均衡化
img_equal = cv2.equalizeHist(img)

cv2.imshow("img_equal",img_equal)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

彩色直方图均衡化

```py
img = cv2.imread("./image/dark1.jpg")
cv2.imshow("src", img)
# 彩色图像均衡化,需要分解通道 对每一个通道均衡化
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
# 合并每一个通道
result = cv2.merge((bH, gH, rH))
cv2.imshow("dst", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
```



### 2.4.2 Log

$$
s=c\log(1+r)
$$

Where c is constant

- 将窄范围的低灰度值映射到更宽范围的输出值，高灰度值则相反
- 扩大暗像素的值或抑制图片中的较高灰度值。

<img src="https://s2.loli.net/2022/11/27/KhwsTyomneaEGOP.png" alt="image-20220920134814625" style="zoom:80%;" />

### 2.4.3 Gamma变换

Gamma变换是对输入图像灰度值进行的非线性操作，使输出图像灰度值与输入图像灰度值呈指数关系：
$$
V_{\rm out}=AV_{\rm in}^{\gamma}
$$
目的：Gamma变换就是用来图像增强，其提升了暗部细节，通过非线性变换，让图像从暴光强度的线性响应变得更接近人眼感受的响应，即将漂白（相机曝光）或过暗（曝光不足）的图片，进行矫正。

gamma 大于 1，图片会更亮，反之变暗

```py
img=cv2.imread('./image/dark1.jpg')
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0/gamma
    table = []
    for i in range(256):
        table.append(((i / 255.0) ** invGamma) * 255)
    table = np.array(table).astype("uint8")
    print(table)
    return cv2.LUT(image, table)

img_gamma = adjust_gamma(img, 0.8)
#print(img_gamma)
cv2.imshow("img",img)
cv2.imshow("img_gamma",img_gamma)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

![image-20220925215514151](https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220925215514151.png)

横坐标是输入灰度值，纵坐标是输出灰度值，蓝色曲线是gamma值小于1时的输入输出关系，红色曲线是gamma值大于1时的输入输出关系。可以观察到，当gamma值小于1时(蓝色曲线)，图像的整体亮度值得到提升，同时低灰度处的对比度得到增加，更利于分辩低灰度值时的图像细节。

### 2.4.3 对比度拉伸

Contrast Stretching

- 最简单的分段线性变换之一
- 增加图像中灰度级的动态范围
- 用于显示设备或重新编码介质以跨越整个灰度范围

$$
\begin{align}
(r_1,s_1)&=(r_2,s_2)\\
r_1&=r_2\\
s_1&=0,s_2=L-1\\
(r_1,s_1)&=(r_\min,0)\\
(r_2,s_2)&=(r_\max,L-1)
\end{align}
$$

当图片的灰度最小值不等于 0 时，或者灰度最大值不等于 255，需要做对比度拉伸

解决图像灰度范围不够广，图像的动态范围

通常是图像处理最后一步

### 2.4.4 灰度反转

黑色图片中有大量明亮细节，做灰度反转，有利于观察



## 2.5 形态学操作

### 2.5.1 图像形态学概要

形态学，是图像处理中应用最为广泛的技术之一，主要用于从图像中提取对表达和描绘区域形状有意义的图像分量，使后续的识别工作能够抓住目标对象最为本质的形状特征，如边界和连通区域等。

结构元素

- 设有两幅图像 B，X
- 若 X（大的图像）是被处理的对象，而 B（小的图像） 是用来处理 X 的，则称 B 为结构元素(structure element)，又被形象地称做刷子。
- 结构元素通常都是一些比较小的图像。  

腐蚀和膨胀

- 图像的膨胀（Dilation）和腐蚀（Erosion）是两种基本的形态学运算，其中
- 膨胀类似于“领域扩张”，将图像中的白色部分进行扩张，其运行结果图比原图的白色区域更大；
- 腐蚀类似于“领域被蚕食”，将图像中白色部分进行缩减细化，其运行结果图比原图的白色区域更小。  

### 2.4.2 图像腐蚀

针对灰度图像

腐蚀的运算符是 “－”，其定义如下：
$$
A-B=\{x|B_x\subseteq A\}
$$
该公式表示图像 A 用卷积模板 B 来进行腐蚀处理，通过模板 B 与图像 A 进行卷积计算，得出 B 覆盖区域的像素点最小值，并用这个最小值来替代参考点的像素值。  

<img src="https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220925220201035.png" alt="image-20220925220201035" style="zoom:67%;" />

把结构元素 B 平移 a 后得到 Ba，若 Ba 包含于X，我们记下这个 a 点，所有满足上述条件的 a 点组成的集合称做 X 被 B 腐蚀(Erosion)的结果。如下图所示。

![image-20220925220235476](https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220925220235476.png)

其中 X 是被处理的对象， B 是结构元素。对于任意一个在阴影部分的点 a， Ba 包含于 X，所以 X 被 B 腐蚀的结果就是那个阴影部分。阴影部分在 X 的范围之内，且比 X 小，就象 X 被剥掉了一层似的。

![image-20220925220242200](https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220925220242200.png)

腐蚀的方法是，拿 B 的中心点和 X 上的点一个一个地对比，如果 B 上的所有点都在 X 的范围内，则该点保留，否则将该点去掉；右边是腐蚀后的结果。可以看出，它仍在原来 X 的范围内，且比X 包含的点要少，就象 X 被腐蚀掉了一层。

---

函数： `cv2.erode(src,element,anchor,iterations)`

- 参数1： `src`，原图像
- 参数2： `element`，腐蚀操作的内核，默认为一个简单的 $3\times3$ 矩形
- 参数3： `anchor`，默认为 `Point(-1,-1)`，内核（结构元素）中心点
- 参数4： `iterations`，腐蚀次数,默认值 1  

腐蚀后的图像

![image-20220925225805658](https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220925225805658.png)

### 2.4.3 图像膨胀

膨胀(dilation)可以看做是腐蚀的对偶运算，其定义是：把结构元素 B 平移 a 后得到 Ba，若 Ba 击中 X，我们记下这个 a 点。所有满足上述条件的 a 点组成的集合称做 X 被 B 膨胀的结果。如下图所示。

![image-20220925225829170](https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220925225829170.png)

其中 X 是被处理的对象， B 是结构元素，对于任意一个在阴影部分的点 a， Ba 击中 X，所以 X 被 B 膨胀的结果就是那个阴影部分。阴影部分包括 X 的所有范围，就像 X 膨胀了一圈似的。  

---

需要说明：若是灰度图像处理时，是寻找结构元素中覆盖目标图像的最大值，赋给结构元素原点位置。（而图中所展示的是最简单的二值图情况下变化）

![image-20220925225938133](https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220925225938133.png)

膨胀时：只要结构元素与目标区域有交集，就保留结构元素的<u>中心点覆盖位置</u>

膨胀后的图像

![image-20220925230006571](https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220925230006571.png)

### 2.4.4 开运算

开运算 = 先腐蚀运算，再膨胀运算（看上去把细微连在一起的两块目标分开了），开运算的效果图如下图所示：  

<img src="https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220925230023813.png" alt="image-20220925230023813"  />

开运算总结：

1. 开运算能够除去孤立的小点，毛刺和小桥，而总的位置和形状不变。
2. 开运算是一个基于几何运算的滤波器。
3. 结构元素大小的不同将导致滤波效果的不同。
4. 不同的结构元素的选择导致了不同的分割，即提取出不同的特征。  

### 2.4.5 闭运算

闭运算 = 先膨胀运算，再腐蚀运算（看上去将两个细微连接的图块封闭在一起），闭运算的效果图如图所示：  

![image-20220925230055009](https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220925230055009.png)

闭运算总结：

1. 闭运算能够填平小湖（即小孔），弥合小裂缝，而总的位置和形状不变。
2. 闭运算是通过填充图像的凹角来滤波图像的。
3. 结构元素大小的不同将导致滤波效果的不同。
4. 不同结构元素的选择导致了不同的分割。  

### 2.4.6 形态学梯度

形态学梯度（Gradient）：

- 基础梯度：基础梯度是用膨胀后的图像减去腐蚀后的图像得到差值图像，也是OpenCV中支持的计算形态学梯度的方法，而此方法得到梯度有称为基本梯度。
- 内部梯度：是用原图像减去腐蚀之后的图像得到差值图像，称为图像的内部梯度。
- 外部梯度：图像膨胀之后再减去原来的图像得到的差值图像，称为图像的外部梯度。  

顶帽和黑帽

- 顶帽（Top Hat）：原图像与开运算图的区别（差值），突出原图像中比周围亮的区域
- 黑帽（Black Hat）：闭操作图像 - 原图像，突出原图像中比周围暗的区域

# 三、图像分割

## 3.1 图像分割

图像分割是指将图像分成若干具有相似性质的区域的过程，主要有**<u>基于阈值</u>**、**<u>基于区域</u>**、**<u>基于边缘</u>**、**<u>基于聚类</u>**、**<u>基于图论</u>**和**<u>基于深度学习</u>**的图像分割方法等。

图像分割分为<u>**语义分割**</u>和**<u>实例分割</u>**。

---

分割的原则就是使划分后的子图在<u>内部保持相似度最大</u>，而<u>子图之间的相似度保持最小</u>。

将 $G = (V,E)$ 分成两个子集 $A$，$B$，使得 $A\cup B=V,\;A\cap B=\empty$

![image-20220926210900640](https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220926210900640.png)

## 3.2 固定阈值

直方图双峰法

- 双峰法：六十年代中期提出的直方图双峰法(也称 mode 法) 是典型的全局单阈值分割方法。
- 基本思想：假设图像中有明显的目标和背景，则其灰度直方图呈双峰分布，当灰度级直方图具有双峰特性时，<u>选取两峰之间的谷对应的灰度级作为阈值</u>。

![image-20220926211039143](https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220926211039143.png)

函数： `cv2.threshold(src, thresh, maxval, type)`

参数说明

- 参数1： 原图像
- 参数2： 对像素值进行分类的阈值
- 参数3： 当像素值高于(小于)阈值时，应该被赋予的新的像素值
- 参数4： 第四个参数是阈值方法。
- 返回值： 函数有两个返回值，一个为 `retVal`, 一个阈值化处理之后的图像。  

<img src="https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220926211123557.png" alt="image-20220926211123557" style="zoom:80%;" />

```py
# 灰度图读入
img = cv2.imread('./image/thresh.png', 0)
threshold = 127
# 阈值分割
ret, th = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
print(ret)

cv2.imshow('thresh', th)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

- `THRESH_BINARY`：$\operatorname{det}(x, y)= \begin{cases}\text { maxval } & \text { if } \operatorname{src}(x, y)>\text { thresh } \\ 0 & \text { otherwise }\end{cases}$
- `THRESH_BINARY_INV`：$\operatorname{dst}(x, y)= \begin{cases}0 & \text { if } \operatorname{arc}(x, y)>\text { thresh } \\ \text { maxval } & \text { otherwise }\end{cases}$
- `THRESH_TRUNC`：$\operatorname{dst}(x, y)= \begin{cases}\operatorname{threshold} & \text { if } \operatorname{src}(x, y)>\operatorname{thresh} \\ \operatorname{src}(x, y) & \text { otherwise }\end{cases}$
- `THRESH_TOZERO`：$\operatorname{dst}(x, y)= \begin{cases}\operatorname{src}(x, y) & \text { if } \operatorname{src}(x, y)>\operatorname{thresh} \\ 0 & \text { otherwise }\end{cases}$
- `THRESH_TOZERO_INV`：$\operatorname{dst}[x, y)= \begin{cases}0 & \text { if } \operatorname{src}(x, y)>\text { thresh } \\ \operatorname{arc}(x, y) & \text { otherwise }\end{cases}$


```py
#opencv读取图像 
img = cv2.imread('./image/person.png',0)
#5种阈值法图像分割
ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img, 127, 255,cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
#使用for循环进行遍历，matplotlib进行显示
for i in range(6):
    plt.subplot(2,3, i+1)
    plt.imshow(images[i],cmap='gray')
    plt.xticks([])
    plt.yticks([])

plt.suptitle('fixed threshold')
plt.show()
```
![image-20220926211420666](https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220926211420666.png)

## 3.3 自动阈值

### 3.3.1 自适应阈值法

函数：`cv2.adaptiveThreshold()`

参数说明：

- 参数1：要处理的原图
- 参数2：最大阈值，一般为255
- 参数3：小区域阈值的计算方式
  - `ADAPTIVE_THRESH_MEAN_C`：小区域内取均值
  - `ADAPTIVE_THRESH_GAUSSIAN_C`：小区域内加权求和，权重是个高斯核
- 参数4：阈值方式（跟前面讲的那5种相同）
- 参数5：小区域的面积，如 11 就是 $11\times11$ 的小块
- 参数6：最终阈值等于小区域计算出的阈值再减去此值

特定： 自适应阈值会每次取图片的一小部分计算阈值，这样图片不同区域的阈值就不尽相同，适用于明暗分布不均的图片。

```py
#自适应阈值与固定阈值对比
img = cv2.imread('./image/paper2.png', 0)

# 固定阈值
ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# 自适应阈值
th2 = cv2.adaptiveThreshold(
    img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11, 4)
th3 = cv2.adaptiveThreshold(
    img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 4)
#全局阈值，均值自适应，高斯加权自适应对比
titles = ['Original', 'Global(v = 127)', 'Adaptive Mean', 'Adaptive Gaussian']
images = [img, th1, th2, th3]
for i in range(4):
    plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i], fontsize=8)
    plt.xticks([]), plt.yticks([])
plt.show()
```

<center class="half">
    <img src="https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220926211706176.png" width="170"/>
    <img src="https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220926211711123.png" width="160"/>
</center>

### 3.3.2 迭代法阈值

步骤

1. 求出图象的最大灰度值和最小灰度值，分别记为 $Z_{\max}$ 和 $Z_{\min}$，令初始阈值 $T_0=(Z_{\max}+Z_{\min})/2$；
2. 根据阈值 $T_K$ 将图象分割为前景和背景，分别求出两者的平均灰度值 $Z_O$ 和 $Z_B$ ；
3. 求出新阈值 $T_{K+1}=(Z_O+Z_B)/2$；
4. 若 $T_K==T_{K+1}$，则所得即为阈值；否则转 2，迭代计算；
5. 使用计算后的阈值进行固定阈值分割。  

### 3.3.3 Otsu大津法

大津法

- 最大类间方差法， 1979年日本学者大津提出，是一种基于<u>全局阈值</u>的自适应方法。
- 灰度特性：图像分为前景和背景。当取最佳阈值时，两部分之间的差别应该是最大的，衡量差别的标准为最大类间方差。
- 直方图有两个峰值的图像，大津法求得的Ｔ近似等于两个峰值之间的低谷。  

<img src="https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220926212030864.png" alt="image-20220926212030864" style="zoom: 67%;" />


$$
\begin{align}
\omega_1&=\frac{N_1}{M \times N} \\
\omega_2&=\frac{N_2}{M \times N} \\
N_1+N_2&=M \times N \\
\omega_1+\omega_2&=1 \\
\mu&=\mu_1 \times \omega_1+\mu_2 \times \omega_2 \\
g&=\omega_1 \times\left(\mu-\mu_1\right)^2+\omega_2 \times\left(\mu-\mu_2\right)^2
\end{align}
$$
得到等价公式
$$
g=\omega_1 \times \omega_2 \times\left(\mu_1-\mu_2\right)^2
$$

等价于最小类内方差 $\omega_1\sigma_1^2+\omega_2\sigma_2^2$

符号说明

- T：图像 $I(x,y)$ 前景和背景的分割阈值
- $\omega_1$：属于前景的像素点数占整幅图像的比例记，其平均灰度 $\mu_1$
- $\omega_2$：背景像素点数占整幅图像的比例为，其平均灰度为 $\mu_2$
- $\mu$：图像的总平均灰度
- $g$：类间方差
- $N_1$：设图像的大小为 $M\times N$，图像中像素的灰度值小于阈值 $T$ 的像素个数
- $N_2$：像素灰度大于阈值 $T$ 的像素个数
- $\sigma$：强度方差

```python
img = cv2.imread('./image/noisy.png', 0)
# 固定阈值法
ret1, th1 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
# Otsu阈值法
ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 先进行高斯滤波，再使用Otsu阈值法，效果最好，实战中一般先进行去噪
blur = cv2.GaussianBlur(img, (5, 5), 0)
ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
images = [img, 0, th1, img, 0, th2, blur, 0, th3]
titles = ['Original', 'Histogram', 'Global(v=100)',
         'Original', 'Histogram', "Otsu's",
         'Gaussian filtered Image', 'Histogram', "Otsu's"]

for i in range(3):
    # 绘制原图
    plt.subplot(3, 3, i * 3 + 1)
    plt.imshow(images[i * 3], 'gray')
    plt.title(titles[i * 3], fontsize=8)
    plt.xticks([]), plt.yticks([])
    
    # 绘制直方图plt.hist, ravel函数将数组降成一维
    plt.subplot(3, 3, i * 3 + 2)
    plt.hist(images[i * 3].ravel(), 256)
    plt.title(titles[i * 3 + 1], fontsize=8)
    plt.xticks([]), plt.yticks([])
    
    # 绘制阈值图
    plt.subplot(3, 3, i * 3 + 3)
    plt.imshow(images[i * 3 + 2], 'gray')
    plt.title(titles[i * 3 + 2], fontsize=8)
    plt.xticks([]), plt.yticks([])
plt.show()
```

![output](https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/output.png)

## 3.4 边缘提取（锐化）

### 3.4.1 图像梯度

梯度

- 梯度是一个向量，梯度方向指向函数变化最快的方向，大小就是它的模，也是最大的变化率，对于二元函数 $z=f(x,y)$，它在点 $(x,y)$ 的梯度就是 ${\rm grad}\,f(x,y)$ 或者 $\bigtriangledown f(x,y)$ 

$$
\text{grad }f(x,y)=\bigtriangledown f(x,y)=(\frac{\partial f}{\partial x},
\frac{\partial f}{\partial y})=
f_x(x,y){\overrightarrow i}+
f_y(x,y){\overrightarrow j}
$$

- 这个梯度向量的幅度和方向角为

$$
\text{mag}(\bigtriangledown f)=||\bigtriangledown f_{(2)}||=\sqrt{G_x^2+G_y^2}\\
\phi(x,y)=\arctan(\frac{G_y}{G_x})
$$

---

图像梯度

- 图像梯度即图像中灰度变化的度量，求图像梯度的过程是二维离散函数求导过程。边缘其实就是图像上灰度级变化很快的点的集合。
- 下表展示了一个灰度图的数学化表达，像素点 $(x,y)$ 的灰度值是 $f(x,y)$，它有八个邻域。

| $f(x-1,y+1)$ | $f(x,y+1)$ | $f(x+1,y+1)$ |
| ------------ | ---------- | ------------ |
| $f(x-1,y)$   | $f(x,y)$   | $f(x+1,y)$   |
| $f(x-1,y-1)$ | $f(x,y-1)$ | $f(x+1,y-1)$ |

- 图像在点 $(x,y)$ 的梯度为

$$
\begin{align}
G(x,y)&=(G_x,G_y)=\left(f_x(x,y),f_y(x,y)\right)\\
f_x(x,y)&=f(x+1,y)-f(x,y)\\
f_y(x,y)&=f(x,y+1)-f(x,y)
\end{align}
$$

- 其中 $f_x(x,y)$ 即 $g_x$ 对应图像的水平方向，
- $g_y(x,y)$ 即 $g_y$ 对应水图像的竖直方向。

---

一阶导

- 均匀变化的色块（梯度在缓慢变化）
- 剧烈变化的起始点
- 检测有无“边”产生

二阶导

- 均匀变化的始末位置
- 孤立点反应强烈
- 剧烈变化的始末位置
- 图像中突然明亮或者暗的点，捕捉细节较多

### 3.4.2 模板卷积

要理解梯度图的生成，就要先了解模板卷积的过程，模板卷积是模板运算的一种方式，其步骤如下

1. 将模板在输入图像中漫游，并将模板中心与图像中某个像素位置重合；
2. 将模板上各个系数与模板下各对应像素的灰度相乘；
3. 将所有乘积相加（为保持灰度范围，常将结果再除以模板系数之和，后面梯度算子模板和为 0 的话就不需要除了）；
4. 将上述运算结果（模板的响应输出）赋给输出图像中对应模板中心位置的像素。  

![image-20220926214116847](https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220926214116847.png)

### 3.4.3 梯度图

梯度图的生成和模板卷积相同，不同的是要生成梯度图，还需要在模板卷积完成后计算在点 $(x,y)$ 梯度的<u>幅值</u>，将幅值作为像素值，这样才算完。

注意： 梯度图上每个像素点的灰度值就是梯度向量的幅度

生成梯度图需要模板，右图为水平和竖直方向最简单的模板。

水平方向：$g(x,y)=|G(x)|=|f(x+1,y)-f(x,y)|$

竖直方向：$g(x,y)=|G(y)|=|f(x,y+1)-f(x,y)|$

<center class="half">
    <img src="https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220926214345034.png" width="180"/>
    <img src="https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220926214349769.png" width="100"/>
    <p>最简单的水平、垂直梯度模板</p>
</center>

### 3.4.5 梯度算子

梯度算子：梯度算子是一阶导数算子，是水平 $G(x)$ 和竖直 $G(y)$ 方向对应模板的组合，也有对角线方向。

常见的一阶算子：Roberts交叉算子， Prewitt算子， Sobel算子  

#### a. Roberts交叉算子

<img src="https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220927114018655.png" alt="image-20220927114018655" style="zoom: 50%;" />

Roberts交叉算子其本质是一个对角线方向的梯度算子，对应的水平方向和竖直方向的梯度分别为
$$
G_x=f(x+1,y+1)-f(x,y)\\
G_y=f(x,y+1)-f(x+1,y)
$$
优点：边缘定位较准，适用于边缘明显且噪声较少的图像。

缺点：

- 没有描述水平和竖直方向的灰度变化，只关注了对角线方向，容易造成遗漏。
- 鲁棒性差。由于点本身参加了梯度计算，不能有效的抑制噪声的干扰。  

#### b. Prewitt算子

<img src="https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220927114133800.png" alt="image-20220927114133800" style="zoom:50%;" />

Prewitt算子是典型的 $3\times3$ 模板，其模板中心对应要求梯度的原图像坐标 $(x,y)$，$(x,y)$ 对应的 8-邻域的像素灰度值如下表所示：

| $f(x-1,y+1)$ | $f(x,y+1)$ | $f(x+1,y+1)$ |
| ------------ | ---------- | ------------ |
| $f(x-1,y)$   | $f(x,y)$   | $f(x+1,y)$   |
| $f(x-1,y-1)$ | $f(x,y-1)$ | $f(x+1,y-1)$ |

通过Prewitt算子的水平模板 $M(x)$ 卷积后，对应的水平方向梯度为

first differentiation in x, smoothing in y
$$
\begin{align}
G_x&=f(x+1,y+1)-f(x-1,y+1)\\
&+f(x+1,y)-f(x-1,y)\\
&+f(x+1,y-1)-f(x-1,y-1)
\end{align}
$$
通过Prewitt算子的竖直模板 $M(y)$ 卷积后，对应的竖直方向梯度为:
$$
\begin{align}
G_y&=f(x-1,y+1)-f(x-1,y-1)\\
&+f(x,y+1)-f(x,y-1)\\
&+f(x+1,y+1)-f(x+1,y-1)
\end{align}
$$
输出梯度图在 $(x,y)$ 的灰度值为:
$$
g(x,y)=\sqrt{G_x^2+G_y^2}
$$
Prewitt算子引入了类似局部平均的运算，对噪声具有平滑作用，较Roberts算子更能抑制噪声。  

#### c. Sobel算子

![image-20220927114720199](https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220927114720199.png)

Sobel算子其实就是是增加了权重系数的Prewitt算子，其模板中心对应要求梯度的原图像坐标，对应的8-邻域的像素灰度值如下表所示

| $f(x-1,y+1)$ | $f(x,y+1)$ | $f(x+1,y+1)$ |
| ------------ | ---------- | ------------ |
| $f(x-1,y)$   | $f(x,y)$   | $f(x+1,y)$   |
| $f(x-1,y-1)$ | $f(x,y-1)$ | $f(x+1,y-1)$ |

---

过Sobel算子的水平模板 $M_{x}$ 卷积后，对应的水平方向梯度为
$$
\begin{align}
G_x&=f(x+1,y+1)-f(x-1,y+1)\\
&+2f(x+1,y)-2f(x-1,y)\\
&+f(x+1,y-1)-f(x-1,y-1)
\end{align}
$$
通过Sobel算子的竖直模板 $M(y)$ 卷积后，对应的竖直方向梯度为

smoothing in x, first differentiation in y
$$
\begin{align}
G_y&=f(x-1,y+1)-f(x-1,y-1)\\
&+2f(x,y+1)-2f(x,y-1)\\
&+f(x+1,y+1)-f(x+1,y-1)
\end{align}
$$
输出梯度图在(x,y)的灰度值为
$$
g(x,y)=\sqrt{G_x^2+G_y^2}
$$
Sobel算子引入了类似局部加权平均的运算，对边缘的定位比要比Prewitt算子好。

---

函数：`dst = cv2.Sobel(src, ddepth, dx, dy, ksize)`

参数说明：

- 参数1：需要处理的图像；
- 参数2：图像的深度， -1 表示采用的是与原图像相同的深度。目标图像的深度必须大于等于原图像的深度；
- 参数3， 4： `dx` 和 `dy` 表示的是求导的阶数， 0 表示这个方向上没有求导，一般为0、 1、 2；
- 参数5：`ksize` 是Sobel算子的大小，必须为1、 3、 5、 7（奇数）。  

```py
img = cv2.imread('image/girl2.png',0)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
# 原图像，位深度，梯度方向，算子大小

plt.subplot(1,3,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.show()
```

![image-20220927115541100](https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220927115541100.png)

## 3.5 Separable filter kernels

![image-20220929130123245](https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220929130123245.png)

允许计算效率更高的实现

![image-20220929130145992](https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220929130145992.png)

## 3.6 Canny边缘检测算法

Canny算子是先平滑后求导数的方法。 John Canny研究了最优边缘检测方法所需的特性，给出了评价边缘检测性能优劣的三个指标

1. 好的信噪比，即将非边缘点判定为边缘点的概率要低，将边缘点判为非边缘点的概率要低；
2. 高的定位性能，即检测出的边缘点要尽可能在实际边缘的中心；
3. 对单一边缘仅有唯一响应，即单个边缘产生多个响应的概率要低，并且虚假响应边缘应该得到最大抑制。  

---

```py
cv2.Canny(image, th1, th2, Size)
```

- image：源图像
- th1：阈值1
- th2：阈值2
- Size：可选参数， Sobel算子的大小

步骤：

1. 彩色图像转换为灰度图像（以灰度图单通道图读入）
2. 对图像进行高斯模糊（去噪）
3. 计算图像梯度，根据梯度计算图像边缘幅值与角度
4. 沿梯度方向进行非极大值抑制（边缘细化）
5. 双阈值边缘连接处理
6. 二值化图像输出结果  

```py
#以灰度图形式读入图像
img = cv2.imread('image/canny.png')
v1 = cv2.Canny(img, 80, 150,(3,3))
v2 = cv2.Canny(img, 50, 100,(5,5))
# 原图，下阈值，上阈值，核大小

#np.vstack():在竖直方向上堆叠
#np.hstack():在水平方向上平铺堆叠
ret = np.hstack((v1, v2))
cv2.imshow('img', ret)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 3.6 连通区域分析算法  

连通区域（Connected Component）一般是指图像中具有相同像素值且位置相邻的前景像素点组成的图像区域，连通区域分析是指将图像中的各个连通区域找出并标记。

连通区域分析是一种在CV和图像分析处理的众多应用领域中较为常用和基本的方法。

例如： OCR识别中字符分割提取（车牌识别、文本识别、字幕识别等）、视觉跟踪中的运动前景目标分割与提取（行人入侵检测、遗留物体检测、基于视觉的车辆检测与跟踪等）、医学图像处理（感兴趣目标区域提取）等。

在需要将前景目标提取出来以便后续进行处理的应用场景中都能够用到连通区域分析方法，通常连通区域分析处理的对象是一张二值化后的图像。 

---

连通区域概要

在图像中，最小的单位是像素，每个像素周围有邻接像素，常见的邻接关系有2种： 4邻接与8邻接。  

<center class="half">
    <img src="https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220927115926429.png" width="180"/>
    <img src="https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220927115929622.png" width="180"/>
</center>

如果A与B连通， B与C连通，则A与C连通，在视觉上看来，彼此连通的点形成了一个区域，而不连通的点形成了不同的区域。这样的一个所有的点彼此连通点构成的集合，我们称为一个连通区域。  

<center class="half">
    <img src="https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220927120101819.png" width="350"/>
    <p>4邻接，则有3个连通区域</p>
    <p>8邻接，则有2个连通区域</p>
</center>

### 3.6.1 Two-Pass 算法

两遍扫描法（ Two-Pass ），正如其名，指的就是通过描两遍图像，将图像中存在的所有连通域找出并标记。

第一次扫描：

- 从左上角开始遍历像素点，找到第一个像素为 255 的点，`label=1`
- 当该像素的左邻像素和上邻像素为无效值时，给该像置一个新的 `label` 值，`label ++`，记录集合
- 当该像素的左邻像素或者上邻像素有一个为有效值时将有效值像素的 `label` 赋给该像素的 `label` 值
- 当该像素的左邻像素和上邻像素都为有效值时，选取其中较小的 `label` 值赋给该像素的 `label` 值 

<center class="half">
    <img src="https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220927120348211.png" width="250"/>
    <img src="https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220927113034430.png" width="250"/>
</center>

第二次扫描：

- 对每个点的label进行更新，更新为其对于其集合中最小的label

<center class="half">
    <img src="https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220927120530714.png" width="250"/>
    <img src="https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220927120534347.png" width="250"/>
</center>

### 3.6.2 区域生长算法 DFS

算法概要

- 区域生长是一种串行区域分割的图像分割方法。区域生长是指从某个像素出发，按照一定的准则，逐步加入邻近像素，当满足一定的条件时，区域生长终止。

区域生长的好坏决定于

- 初始点（种子点）的选取。
- 生长准则。
- 终止条件。

区域生长是从某个或者某些像素点出发，最后得到整个区域，进而实现目标的提取。  

---

原理

- 基本思想：将具有相似性质的像素集合起来构成区域。

步骤：

1. 对图像顺序（随机）扫描，找到第 1 个还没有归属的像素, 设该像素为 $(x_0, y_0)$
2. 以 $(x_0, y_0)$ 为中心, 考虑 $(x_0, y_0)$ 的4邻域像素 $(x, y)$ 如果 $(x_0, y_0)$ 满足生长准则，将 $(x, y)$ 与 $(x_0, y_0)$ 合并(在同一区域内)，同时将 $(x, y)$ 压入堆栈
3. 从堆栈中取出一个像素, 把它当作 $(x_0, y_0)$ 返回到步骤 2
4. 当堆栈为空时，返回到步骤 1
5. 重复步骤 1 - 4 直到图像中的每个点都有归属时
6. 生长结束。  

![image-20220927120916606](https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220927120916606.png)

## 3.7 分水岭算法

任意的灰度图像可以被看做是地质学表面，高亮度的地方是山峰，低亮度的地方是山谷。 

![image-20220927120939697](https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220927120939697.png)

算法概要

- 给每个孤立的山谷（局部最小值）不同颜色的水（标签），当水涨起来，根据周围的山峰（梯度），不同的山谷也就是不同的颜色会开始合并，要避免山谷合并，需要在水要合并的地方建立分水岭，直到所有山峰都被淹没，所创建的分水岭就是分割边界线，这个就是分水岭的原理。  

---

算法步骤

1. 加载原始图像
2. 阔值分割，将图像分割为黑白两个部分
3. 对图像进行开运算，即先腐蚀在膨胀
4. 对开运算的结果再进行膨胀，得到大部分是真实背景的区域（远离目标的区域）
5. 通过距离变换 Distance Transform 获取前景区域
6. 背景区域 `sure_bg` 和前景区域 `sure_fg` 相减，得到即有前景又有背景的重合区域，即边界
7. 连通区域处理，标记边界区域
8. 得到分水岭算法结果

## 3.8 Meanshift

传入图片，图片每个点看成数据，数据做聚类，分割就出来了

聚类算法一般 k-means，先验知识知道 k 值

# 四、图像特征与目标检测

## 4.1 图像特征理解

图像特征是图像中独特的， 易于跟踪和比较的特定模板或特定结构。

特征就是有意义的图像区域， 该区域具有独特性或易于识别性！  

图像特征提取与匹配是计算机视觉中的一个关键问题， 在目标检测、物体识别、 三维重建、 图像配准、 图像理解等具体应用中发挥着重要作用。

图像特征主要有图像的颜色特征、 纹理特征、 形状特征和空间关系特征。

---

颜色特征

- 颜色特征是一种全局特征， 描述了图像或图像区域所对应的景物的表面性质
- 颜色特征描述方法：颜色直方图、颜色空间、颜色分布  

纹理特征

- 纹理特征也是一种全局特征，它也描述了图像或图像区域所对应景物的表面性质。但由于纹理只是一种物体表面的特性，并不能完全反映出物体的本质属性，所以仅仅利用纹理特征是无法获得高层次图像内容的。

形状特征

- 形状特征有两类表示方法，一类是轮廓特征， 另一类是区域特征。
- 图像的轮廓特征主要针对物体的外边界，而图像的区域特征则描述了是图像中的局部形状特征。  

空间关系特征

- 是指图像中分割出来的多个目标之间的相互的空间位置或相对方向关系
- 这些关系也可分为连接/邻接关系、 交叠/重叠关系和包含/独立关系等。  

## 4.2 形状特征描述

### 4.2.1 HOG 特征

特征提取

- 方向梯度直方图（Histogram of Oriented Gradient, HOG）特征是一种在计算机视觉和图像处理中用来进行物体检测的特征描述子。
- 它通过计算和统计图像局部区域的梯度方向直方图来构成特征。
- Hog特征结合SVM分类器已经被广泛应用于图像识别中， 尤其在行人检测中获得了极大的成功。
- 主要思想： 在一副图像中， 目标的形状能够被梯度或边缘的方向密度分布很好地描述  

---

实现过程

- 灰度化（将图像看做一个 $x,y,z$（灰度）的三维图像）；
- 采用 Gamma 校正法对输入图像进行颜色空间的标准化（归一化）；
- 计算图像每个像素的梯度（包括大小和方向）；
- 将图像划分成小 cells；
- 统计每个cell的梯度直方图（不同梯度的个数），得到cell的描述子；
- 将每几个 cell 组成一个 block，得到block的描述子（descriptor）；
- 将图像image内的所有block的HOG特征descriptor串联起来就可以得到HOG特征，该==特征向量==就是用来目标检测或分类的特征

---

角点概念

- 角点： 在现实世界中， 角点对应于物体的拐角， 道路的十字路口、 丁字路口等。
- 从图像分析的角度来定义角点可以有以下两种定义：
  - 角点可以是两个边缘的交点；
  - 角点是邻域内具有两个主方向的特征点；
- 角点计算方法：
  - 前者通过图像边缘计算，计算量大， 图像局部变化会对结果产生较大的影响；
  - 后者基于图像灰度的方法通过计算点的曲率及梯度来检测角点  

### 4.2.2 Harris 角点检测

角点：在现实世界中， 角点对应于物体的拐角， 道路的十字路口、 丁字路口等。从图像分析的角度来定义角点可以有以下两种定义：

- 角点可以是两个边缘的交点；
- 角点是邻域内具有两个主方向的特征点；

角点计算方法：

- 前者通过图像边缘计算， 计算量大， 图像局部变化会对结果产生较大的影响；
- 后者基于图像灰度的方法通过计算点的曲率及梯度来检测角点；  

角点所具有的特征

- 轮廓之间的交点
- 对于同一场景， 即使视角发生变化， 通常具备稳定性质的特征；
- 该点附近区域的像素点无论在梯度方向上还是其梯度幅值上有着较大变化；

性能较好的角点

- 检测出图像中“真实” 的角点
- 准确的定位性能
- 很高的重复检测率
- 噪声的鲁棒性
- 较高的计算效率  

---

Harris实现过程

-  计算图像在X和Y方向的梯度
  - $I_x=\frac{\partial I}{\partial x}=I\otimes(-1\;0\;1)$
  - $I_y=\frac{\partial I}{\partial y}=I\otimes(-1\;0\;1)^T$
- 计算图像两个方向梯度的乘积
  - $I^2_x=I_x\cdot I_x$
  - $I^2_y=I_y\cdot I_y$
  - $I_{xy}=I_x\cdot I_y$
- 使用高斯函数对三者进行高斯加权，生成矩阵 $M$ 的 $A$，$B$，$C$；
  - $A=g(I_x^2)=I_x^2\otimes w$
  - $C=g(I_y^2)=I_y^2\otimes w$
  - $B=g(I_{xy})=I_{xy}\otimes w$
- 计算每个像素的Harris响应值 $R$， 并对小于某一阈值 $t$ 的 $R$ 置为零；
- 在 $3\times 3$ 或 $5\times 5$ 的邻域内进行非最大值抑制， 局部最大值点即为图像中的角点；  

---

Harris代码应用

Open 中的函数 `cv2.cornerHarris()` 可以用来进行角点检测。 参数如下:

- `img` - 数据类型为 `float32` 的输入图像
- `blockSize` - 角点检测中要考虑的领域大小
- `ksize` - Sobel 求导中使用的窗口大小
- `k` - Harris 角点检测方程中的自由参数，取值参数为 `[0,04,0.06]`

```py
filename = 'harris2.png'
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

# 输入图像必须是 float32 ,最后一个参数在 0.04 到 0.06 之间
dst = cv2.cornerHarris(gray,2,3,0.06)

#结果进行膨胀，可有可无
dst = cv2.dilate(dst,None)
print(dst)
# 设定阈值，不同图像阈值不同
img[dst>0.01*dst.max()]=[0,0,255]
```

### 4.2.3 SIFT算法

SIFT，即尺度不变特征变换算法（Scale-invariant feature transform，SIFT），是用于图像处理领域的一种算法。SIFT具有尺度不变性，可在图像中检测出关键点，是一种局部特征描述子。

其应用范围包含物体辨识、 机器人地图感知与导航、 影像缝合、 3D模型建立、手势辨识、 影像追踪和动作比对  

---

SIFT实现过程

- SIFT特性
  - 独特性，也就是特征点可分辨性高， 类似指纹， 适合在海量数据中匹配。
  - 多量性，提供的特征多。
  - 高速性，就是速度快。
  - 可扩展，能与其他特征向量联合使用。  
- SIFT特点
  - 旋转、 缩放、 平移不变性
  - 解决图像仿射变换， 投影变换的关键的匹配
  - 光照影响小
  - 目标遮挡影响小
  - 噪声景物影响小

---

SIFT算法步骤

- 尺度空间极值检测点检测
- 关键点定位： 去除一些不好的特征点， 保存下来的特征点能够满足稳定性等条件
- 关键点方向参数： 获取关键点所在尺度空间的邻域， 然后计算该区域的梯度和方向， 根据计算得到的结果创建方向直方图， 直方图的峰值为主方向的参数
- 关键点描述符： 每个关键点用一组向量（位置、 尺度、 方向） 将这个关键点描述出来， 使其不随着光照、 视角等等影响而改变
- 关键点匹配：分别对模板图和实时图建立关键点描述符集合， 通过对比关键点描述符来判断两个关键点是否相同  

---

SIFT代码实现（输入是一张图片，输出是n个向量）

- 返回的关键点是一个带有很多不用属性的特殊结构体， 属性当中有坐标， 方向、 角度等。
- 使用 `sift.compute()` 函数来进行计算关键点描述符
  - `kp,des = sift.compute(gray,kp)`
- 如果未找到关键点， 可使用函数 `sift.detectAndCompute()` 直接找到关键点并计算。
- 在第二个函数中， `kp` 为关键点列表， `des` 为 numpy 的数组， 为关键点数目$\times 128$  

```py
img = cv2.imread('harris2.png')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create() # 定义sift 取器
kp = sift.detect(gray,None)#找到关键点
img=cv2.drawKeypoints(gray,kp,img)#绘制关键点
cv2.imshow('sp',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

#### 特征编码 bag of words

局部 SIFT 特征的全局编码

- 将图像的局部特征（SIFT 关键点描述符）集成到全局矢量中以表示整个图像

SIFT 算法能将一副图片转变成一个有意义的向量吗？不能

- 转换成了一组向量

不同的图片，通过同一个 SIFT 算法去提取特征点，最后得到的特征数量是**不一样**的

---

本周：将一幅图片转换成个月有意义的向量

- SIFT 算法将图片转换成一组有意义的向量
- BoW 将一组有意义的向量转变成一个有意义的向量

---

最流行的方法：词袋模型（BoW）

- 将可变数量的本地图像特征编码为固定维度直方图以表示每个图像

<img src="https://s2.loli.net/2022/12/05/buhOpV6fTdwW3BQ.png" alt="image-20221003194415902" style="zoom:80%;" />

<img src="https://s2.loli.net/2022/12/05/h9WjyoFVL8EQv3d.png" alt="image-20221205134516170" style="zoom:80%;" />

<img src="https://s2.loli.net/2022/12/05/s6lEDoyfBjYHKwN.png" alt="image-20221205134534496" style="zoom:80%;" />

<img src="https://s2.loli.net/2022/12/05/ljmEHpguYP8wWNc.png" alt="image-20221003195029024" style="zoom:80%;" />

## 4.3 纹理特征

### 4.3.1 LBP

LBP（Local Binary Pattern， 局部二值模式），是一种用来描述图像局部纹理特征的算子；它具有旋转不变性和灰度不变性等显著的优点

---

LBP原理

- LBP算子定义在一个 $3\times 3$ 的窗口内， 以窗口中心像素为阈值， 与相邻的 8 个像素的灰度值比较， 若周围的像素值大于中心像素值， 则该位置被标记为1;， 否则标记为 0。
- 如此可以得到一个 8 位二进制数（通常还要转换为 10 进制，即 LBP 码，共256 种），将这个值作为窗口中心像素点的 LBP 值，以此来反应这个 $3\times3$ 区域的纹理信息。

![image-20220928172257819](https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220928172257819.png)

数学公式
$$
\text{LBP}(x_c,y_c)=\sum\limits_{p=1}^8 s(I(p)-I(c))\times 2^p
$$

- 其中，$p$ 表示 $3\times  3$ 窗口中除中心像素点外的第 $p$ 个像素点；

- $I(c)$ 表示中心像素点的灰度值，$I(p)$ 表示领域内第 $p$ 个像素点的灰度值；

- $s(x)$ 公式如下

$$
s(x) = \left\{ {\begin{array}{*{20}{c}}
{1}&{x \ge 0}\\
{0}&{{\rm{otherwise}}}
\end{array}} \right.
$$
- LBP记录的是中心像素点与领域像素点之间的差值
- 当光照变化引起像素灰度值同增同减时， LBP 变化并不明显
- LBP对与光照变化不敏感， LBP 检测的仅仅是图像的纹理信息

### 4.3.2 Haralick Features

步骤 1 构造 GLCMs 灰度共生矩阵 – 给定方向角 $\theta$ 处的距离 $d$

- 计算从灰度级别 $l_1$ 到另一个灰度级别 $l_2$ 的共现计数（或概率），沿轴的采样间间距为 $d$，使角度 $\theta$ 与 $x$ 轴成角度 $\theta$ 并将其表示为 $p_{(d,\theta)}(l_1,l_2)$
- 构造矩阵 $P(d,\theta)$，元素 $(l_1,l_2)$ 为 $p_{(d,\theta)}(l_1,l_2)$
- 如果量子化图像中不同灰度级的数量为 $L$，则共发矩阵 $P$ 的大小为 $L\times L$

---

- 为了提高计算效率，可以通过合并（类似于直方图合并）来减少灰度级（L）的数量，例如 $L = 256 / n$，$n$ 是常数因子。
- 通过使用距离（$d$）和角方向（$\theta$）的各种组合，可以构造不同的共生矩阵。
- 就其本身而言，这些共生矩阵不提供任何可以很容易地用作纹理描述符的纹理度量。
- 共生矩阵中的信息需要进一步提取为一组特征值 =>  Haralick descriptor。

---

步骤 2：从 GLCMs 计算 Haralick descriptor

- 每个 GLCM 对应一组 Haralick descriptor，对应于特定距离 （d） 和角方向 （$\theta$）

![image-20221003140047520](https://s2.loli.net/2022/12/05/UfgswLZjGRapTAl.png)

在灰度共生矩阵的基础上，一个灰度共生矩阵，通过上面公式计算一个标量



  

## 4.4 模板匹配

模板匹配介绍

- 模板匹配是一种最原始、最基本的模式识别方法，研究某一特定目标的图像位于图像的什么地方，进而对图像进行定位
- 在待检测图像上，从左到右，从上向下计算模板图像与重叠子图像的匹配度，匹配程度越大，两者相同的可能性越大

![image-20220928173048986](https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220928173048986.png)

函数：`result=cv2.matchTemplate(image, templ, method)`

- `image` 参数表示待搜索图像
- `templ` 参数表示模板图像，必须不大于源图像并具有相同的数据类型。
- `method` 参数表示计算匹配程度的方法。

匹配方法

- `TM_SQDIFF_NORMED` 是标准平方差匹配，通过计算两图之间平方差来进行匹配，最好匹配为0，匹配越差，匹配值越大。
- `TM_CCORR_NORMED` 是标准相关性匹配，采用模板和图像间的乘法操作，数越大表示匹配程度较高，0表示最坏的匹配效果，这种方法除了亮度线性变化对相似度计算的影响。
- `TM_CCOEFF_NORMED` 是标准相关性系数匹配，两图减去了各自的平均值之外，还要各自除以各自的方差。将模版对其均值的相对值与图像对其均值的相关值进行匹配，1表示完美匹配，-1表示糟糕的匹配，0表示没有任何相关性（随机序列）。

---

对模板匹配得到的结果进行计算

函数：`minVal, maxVal, minLoc, maxLoc=cv2.minMaxLoc()`

- `minVal` 参数表示返回的最小值（匹配得到的最小值）
- `maxVal` 参数表示返回的最大值
- `minLoc` 参数表示返回的最小位置
- `maxLoc` 参数表示返回的最大位置

## 4.5 人脸检测

人脸识别概要

- 一般而言，一个完整的人脸识别系统包含四个主要组成部分，即人脸检测、人脸对齐、人脸特征提取以及人脸识别。
- 四部分流水线操作：
  - 人脸检测在图像中找到人脸的位置；
  - 人脸配准在人脸上找到眼睛、鼻子、嘴巴等面部器官的位置；
  - 通过人脸特征提取将人脸图像信息抽象为字符串信息；
  - 人脸识别将目标人脸图像与既有人脸比对计算相似度，确认人脸对应的身份。

#### 4.5.1 人脸检测

- 人脸检测算法的输入是一张图片，输出是人脸框坐标序列（0个人脸框或1个人脸框或多个人脸框）。一般情况下，输出的人脸坐标框为一个正朝上的正方形，但也有一些人脸检测技术输出的是正朝上的矩形，或者是带旋转方向的矩形。

### 4.5.2 人脸对齐

Face Alignment

- 根据输入的人脸图像，自动定位出人脸上五官关键点坐标的一项技术
- 人脸对齐算法的输入是“一张人脸图片”加“人脸坐标框”，输出五官关键点的坐标序列。五官关键点的数量是预先设定好的一个固定数值，可以根据不同的语义来定义（常见的有5点、68点、90点等等）。
- 对人脸图像进行特征点定位，将得到的特征点利用仿射变换进行人脸矫正，若不矫正，非正面人脸进行识别准确率不高。

### 4.5.3 人脸特征提取

Face Feature Extraction

- 将一张人脸图像转化为一串固定长度的数值的过程
- 具有表征某个人脸特点能力的数值串被称为 “人脸特征（Face Feature）”。

<img src="https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220930114425901.png" alt="image-20220930114425901" style="zoom: 67%;" />

### 4.5.4 人脸识别（Face Recognition）

- 识别出输入人脸图对应身份的算法
- 输入一个人脸特征，通过和注册在库中 N 个身份对应的特征进行逐个比对，找出“一个”与输入特征相似度最高的特征。
- 将这个最高相似度值和预设的阀值相比较，如果大于阈值，则返回该特征对应的身份，否则返回“不在库中”。

<img src="https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220930114625627.png" alt="image-20220930114625627" style="zoom:80%;" />

# 五、运动目标识别

## 5.1 摄像头调用

开启摄像头

- 函数1：`cv2.VideoCapture()`
  - 参数说明： 0,1 代表电脑摄像头， 或视频文件路径。
- 函数2：`ret,frame = cap.read()`
- 说明：`cap.read()` 按帧读取视频
  - `ret`：返回布尔值 True/False，如果读取帧是正确的则返回 True，如果文件读取到结尾，它的返回值就为 False；
  - `frame`： 每一帧的图像， 是个三维矩阵  

下面的程序将使用 OpenCV 调用摄像头， 并实时播放摄像头中画面，按下 “q”  键结束播放  

```py
cap = cv2.VideoCapture(0)
while(True):
    #获取一帧帧图像
    ret, frame = cap.read()
    #转化为灰度图
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',frame)
    #按下“q”键停止
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release() # 释放摄像头
cv2.destroyAllWindows()
```

## 5.2 播放、保存视频

### 5.2.1 保存视频

- 指定写入视频帧编码格式
- 函数 `fourcc = cv2.VideoWriter_fourcc('M','J','P','G')`

![image-20220930120623620](https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220930120623620.png)

### 5.2.2 保存视频

创建 `VideoWriter` 对象

函数 `out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))`

参数说明：

- 参数1： 保存视频路径 + 名字；
- 参数2： `FourCC` 为4 字节码， 确定视频的编码格式；
- 参数3： 播放帧率
- 参数4： 大小
- 参数5： 默认为 True， 彩色图  

```py
#调用摄像头函数cv2.VideoCapture，参数0：系统摄像头
cap = cv2.VideoCapture(0)
#创建编码方式
#mp4:'X','V','I','D'avi:'M','J','P','G'或'P','I','M','1' flv:'F','L','V','1'
fourcc = cv2.VideoWriter_fourcc('F','L','V','1')

#创建VideoWriter对象
out = cv2.VideoWriter('output_1.flv',fourcc, 20.0, (640,480))
```

## 5.3 帧差法目标识别

帧差法

- 帧间差分法是通过对视频中相邻两帧图像做差分运算来标记运动物体的方法。
- 当视频中存在移动物体的时候， 相邻帧（或相邻三帧）之间在灰度上会有差别，求取两帧图像灰度差的绝对值，则静止的物体在差值图像上表现出来全是0，而移动物体特别是移动物体的轮廓处由于存在灰度变化为非0。  

优点

- 算法实现简单，程序设计复杂度低；
- 对光线等场景变化不太敏感， 能够适应各种动态环境， 稳定性较好；

缺点

- 不能提取出对象的完整区域，对象内部有“空洞” ；
- 只能提取出边界，边界轮廓比较粗，往往比实际物体要大；
- 对快速运动的物体，容易出现糊影的现象，甚至会被检测为两个不同的运动物体；
- 对慢速运动的物体，当物体在前后两帧中几乎完全重叠时，则检测不到物体；  （可以考虑跳帧）

## 5.4 光流法

光流法利用图像序列中像素在时间域上的变化以及相邻帧之间的相关性，根据上一帧与当前帧之间的对应关系， 计算得到相邻帧之间物体的运动信息。

大多数的光流计算方法计算量巨大， 结构复杂， 且易受光照、 物体遮挡或图像噪声的影响，鲁棒性差，故一般不被对精度和实时性要求比较高的监控系统所采用

光流是基于以下假设的

- 在连续的两帧图像之间（目标对象的）像素的灰度值不改变。
- 相邻的像素具有相同的运动

## 5.5 背景减除法

背景消除

- OpenCV中常用的两种背景消除方法， 一种是基于高斯混合模型GMM实现的背景提取， 另外一种是基于最近邻KNN实现的。

![image-20220930121033648](https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20220930121033648.png)

---

GMM模型

- MOG2 算法， 高斯混合模型分离算法， 它为每个像素选择适当数量的高斯分布， 它可以更好地适应不同场景的照明变化等
- 函数： `cv2.createBackgroundSubtractorMOG2(int history = 500, double varThreshold = 16,bool detectShadows = true )`

KNN模型

- `cv2.createBackgroundSubtractorKNN()`

---

方法

- 主要通过视频中的背景进行建模， 实现背景消除， 生成 mask 图像， 通过对 mask 二值图像分析实现对前景活动对象的区域的提取， 整个步骤如下：
- 1. 初始化背景建模对象 GMM
  2. 读取视频一帧
  3. 使用背景建模消除生成 mask
  4. 对mask 进行轮廓分析提取 ROI（region of interest）
  5. 绘制 ROI 对象 

