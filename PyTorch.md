[toc]

# 一、张量

Variable 是 `torch.autograd` 中的数据类型

主要用于封装 Tensor，进行自动求导

- `data`：被包装的Tensor
- `grad`：data 的梯度
- `grad_fn`：创建 Tensor 的 Function，是自动求导的关键
- `requires_ grad`：指示是否需要梯度
- `is_leaf`：指示是否是叶子结点（张量）  

PyTorch0.4.0 版开始，Variable 并入 Tensor

- `dtype`：张量的数据类型，如 `torch.FloatTensor,torch.cuda.FloatTensor`
- `shape`：张量的形状，如 `(64,3,224,224)`
- `device`：张量所在设备，GPU/CPU，是加速的关键

## 1.1 创建

### 1.1.1 直接创建

```py
torch.tensor(data,
            dtype=None,
            device=None,
            requires_grad=False,
            pin_memory=False)
```

功能：从 data 创建 tensor

- `data`: 数据, 可以是 list, numpy
- `dtype`：数据类型，默认与data的一致
- `device`：所在设备 cuda/cpu
- `requires_grad`：是否需要梯度
- `pin_memory`：是否存于锁页内存

```py
torch.from_numpy(ndarray)
```

功能：从 numpy 创建 tensor

注意事项：从 `torch.from_numpy` 创建的 tensor 于原 `ndarray` 共享内存，当修改其中一个的数据，另外一个也将会被改动 

### 1.1.2 依据数值创建

```py
torch.zeros(*size,
            out=None,
            dtype=None,
            layout=torch.strided,
            device=None,
            requires_grad=False)
```

功能：依 `size` 创建全 0 张量

- `size`：张量的形状，如 (3,3)、(3,224,224)
- `out`：输出的张量
- `layout`:内存中布局形式,有 `strided`，`sparse_coo` 等
- `device`：所在设备 gpu/cpu
- `requires_grad`：是否需要梯度

---

```py
torch.zeros_like(input,
                dtype=None,
                layout=None,
                device=None,
                requires_grad=False)
```

功能：依 input 形状创建全 0 张量

- `input`：创建与 `input` 同形状的全 0 张量
- `dtype`：数据类型
- `layout`：内存中布局形式

---

```py
torch.ones(*size,
            out=None,
            dtype=None,
            layout=torch.strided,
            device=None,
            requires_grad=False)
           
torch.ones_like(input,
            dtype=None,
            layout=None,
            device=None,
            requires_grad=False)
```

功能：依input形状创建全1张量

- `size`：张量的形状,如(3,3)、(3,224,224)
- `dtype`：数据类型
- `layout`：内存中布局形式
- `device`：所在设备 gpu/cpu
- `requires_grad`：是否需要梯度

---

```py
# torch.full_like()
torch.full(size,
            fill_value,
            out=None,
            dtype=None,
            layout=torch.strided,
            device=None,
            requires_grad=False)
```

功能：依 input 形状创建指定数据的张量

- `size`：张量的形状,如(3,3)
- `fill_value`：张量的值

```py
torch.arange(start=0, end, step=1,
             out=None,
             dtype=None,
             layout=torch.strided,
             device=None,
             requires_grad=False)
```

功能：创建等差的 1 维张量

注意事项：数值区间为 `[start,end)`

- `start`：数列起始值
- `end`：数列 “结束值”
- `step`：数列公差，默认为 1

---

```py
torch.linspace(start, end, steps=100,
                out=None,
                dtype=None,
                layout=torch.strided,
                device=None,
                requires_grad=False)
```

功能：创建均分的 1 维张量

注意事项：数值区间为 `[start,end]`

- `start`：数列起始值
- `end`：数列结束值
- `steps`：数列长度

---

```py
torch.logspace(start, end, steps=100,
                base=10.0,
                out=None,
                dtype=None,
                layout=torch.strided,
                device=None,
                requires_grad=False)
```

功能：创建对数均分的 1 维张量

注意事项：长度为`steps`，底为 `base`

- `start`：数列起始值
- `end`：数列结束值
- `steps`：数列长度
- `base`：对数函数的底，默认为10

---

```py
torch.eye(n,
          m=None,
          out=None,
          dtype=None,
          layout=torch.strided,
          device=None,
          requires_grad=False)
```

功能：创建单位对角矩阵（2维张量）

注意事项：默认为方阵

- n：矩阵行数
- m：矩阵列数

### 1.1.3 依概率分布创建张量

```py
torch.normal(mean, std, size, out=None)
```

功能：生成正态分布（高斯分布）

- `mean`：均值
- `std`：标准差

四种模式：

- `mean` 为标量，`std` 为标量
- `mean` 为标量，`std` 为张量
- `mean` 为张量，`std` 为标量
- `mean` 为张量，`std` 为张量

---

```py
# 生成标准正态分布
torch.randn(*size, # 张量的形状
            out=None,
            dtype=None,
            layout=torch.strided,
            device=None,
            requires_grad=False)

torch.randn_like()

# 在区间 [0, 1) 上，生成均匀分布
torch.rand(*size,
            out=None,
            dtype=None,
            layout=torch.strided,
            device=None,
            requires_grad=False)
torch.rand_like()

# 区间[low, high) 生成整数均匀分布
torch.randint(low=0, high,
              size, # 张量的形状
              out=None,
              dtype=None,
              layout=torch.strided,
              device=None,
              requires_grad=False)
```



```py
# 依概率分布创建张量

# 生成从 0 到 n-1 的随机排列
torch.randperm(n, # 张量的长度
               out=None,
               dtype=torch.int64,
               layout=torch.strided,
               device=None,
               requires_grad=False)

# 以 input 为概率，生成伯努力分布（ 0-1 分布，两点分布）
torch.bernoulli(input, # 概率值
                *,
                generator=None,
                out=None)
```

## 1.2 张量操作

### 1.2.1 拼接与切分

```py
torch.cat(tensors, dim=0, out=None)
```

将张量按维度 dim 进行拼接

- tensors：张量序列
- dim：要拼接的维度

---

```py
torch.stack(tensors, dim=0, out=None)
```

在 **<u>新创建的维度</u>** dim 上进行拼接

- `tensors`：张量序列
- `dim`：要拼接的维度  

---

```py
torch.chunk(input, # 要切分的张量
            chunks, # 要切分的份数
            dim=0) # 要切分的维度
```

功能：将张量按维度 dim 进行平均切分

返回值：张量列表

注意事项：若不能整除，最后一份张量小于其他张量

---

```py
torch.split(tensor, # 要切分的张量
            split_size_or_sections, # 为int时，表示每一份的长度；为list时，按list元素切分
            dim=0) # 要切分的维度
```

### 1.2.2 索引

```py
torch.index_select(input, # 要索引的张量
                    dim, # 要索引的维度
                    index, # 要索引数据的序号
                    out=None)
```

功能：在维度 dim 上，按 index 索引数据

返回值：依 index 索引数据拼接的张量  

---

```py
torch.masked_select(input, # 要索引的张量
                    mask, # 与input同形状的布尔类型张量  
                    out=None)
```

功能：按 mask 中的 True 进行索引

返回值：一维张量
---

```py
torch.reshape(input, # 要变换的张量
              shape) # 新张量的形状
```

功能：变换张量形状

注意事项：当张量在内存中是连续时，新张量与 input 共享数据内存  

---

```py
# 交换张量的两个维度
torch.transpose(input, # 要变换的张量
                dim0, # 要交换的维度
                dim1) # 要交换的维度

torch.t(input) #  2维张量转置，对矩阵而言，等价于 torch.transpose(input, 0, 1) 
```



```py
# 压缩长度为1的维度（轴）
torch.squeeze(input,
              dim=None, # 若为None，移除所有长度为1的轴；若指定维度，当且仅当该轴长度为1时，可以被移除
              out=None)

# 依据dim扩展维度
torch.unsqueeze(input,
               dim, # 扩展的维度
               out=None)
```



### 1.2.3 数学运算  



```py
# 加减乘除
torch.add()
torch.addcdiv()
torch.addcmul()
torch.sub()
torch.div()
torch.mul()

# 对数，指数，幂函数
torch.log(input, out=None)
torch.log10(input, out=None)
torch.log2(input, out=None)
torch.exp(input, out=None)
torch.pow()

# 三角函数
torch.abs(input, out=None)
torch.acos(input, out=None)
torch.cosh(input, out=None)
torch.cos(input, out=None)
torch.asin(input, out=None)
torch.atan(input, out=None)
torch.atan2(input, other, out=None)
```



```py
torch.add(input,
          alpha=1,
          other,
          out=None)
```

功能：逐元素计算 $\rm input+alpha\times other$

• `input`：第一个张量

• `alpha`：乘项因子

• `other`：第二个张量  

```py
torch.addcmul(input,
              value=1,
              tensor1,
              tensor2,
              out=None)
```

- `torch.addcdiv()`：$\rm out_i=input_i+value\times \frac{tensor1_i}{tensor2_2}$
- `torch.addcmul()`：$\rm out_i = input_i+value\times tensor1_i\times tensor2_i$

# 二、`autograd`

```py
torch.autograd.backward(tensors,
                        grad_tensors=None,
                        retain_graph=None,
                        create_graph=False)
```

功能：自动求取梯度

- `tensors`：用于求导的张量，如 loss
- `retain_graph`：保存计算图
- `create_graph`：创建导数计算图，用于高阶求导
- `grad_tensors`：多梯度权重

`autograd` 小贴士：

1. 梯度不自动清零
2. 依赖于叶子结点的结点， `requires_grad` 默认为 `True`
3. 叶子结点不可执行 `in-place`

# 三、`DataLoader` & `Dataset`

## 3.1 `DataLoader`

```py
# torch.utils.data.DataLoader
DataLoader(dataset,
           batch_size=1,
           shuffle=False,
           sampler=None,
           batch_sampler=None,
           num_workers=0,
           collate_fn=None,
           pin_memory=False,
           drop_last=False,
           timeout=0,
           worker_init_fn=None,
           multiprocessing_context=None)
```

功能：构建可迭代的数据装载器

- `dataset`：`Dataset` 类，决定数据从哪读取及如何读取
- `batchsize`：批大小
- `num_works`：是否多进程读取数据
- `shuffle`：每个 `epoch` 是否乱序
- `drop_last`：当样本数不能被 `batchsize` 整除时，是否舍弃最后一批数据

---

`Epoch`：所有训练样本都已输入到模型中，称为一个 Epoch

`Iteration`：一批样本输入到模型中，称之为一个 Iteration

`Batchsize`：批大小，决定一个 Epoch 有多少个 Iteration

样本总数：80， `Batchsize`：8

- 1 Epoch = 10 Iteration

样本总数：87， Batchsize：8

- 1 Epoch = 10 Iteration ？ `drop_last = True`
- 1 Epoch = 11 Iteration ？ `drop_last = False`

## 3.2 Dataset

```py
class Dataset(object):
    def __getitem__(self, index):
    	raise NotImplementedError
    def __add__(self, other):
    	return ConcatDataset([self, other])
```

功能： Dataset 抽象类，所有自定义的 Dataset 需要继承它，并且复写 `__getitem__()`

`getitem`

- 接收一个索引，返回一个样本  

<img src="https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20221006192016277.png" alt="image-20221006192016277" style="zoom: 67%;" />

<img src="https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20221006192135983.png" alt="image-20221006192135983" style="zoom:80%;" />

# 四、transforms

- `torchvision.transforms`：常用的图像预处理方法
- `torchvision.datasets`：常用数据集的 dataset 实现， MNIST， CIFAR-10， ImageNet等
- `torchvision.model`：常用的模型预训练， AlexNet， VGG， ResNet， GoogLeNet等  

<img src="https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20221006195256504.png" alt="image-20221006195256504" style="zoom:80%;" />

```py
transforms.Normalize(mean, # 各通道的均值
                     std, # 各通道的标准差
                     inplace=False) # 是否原地操作
```

功能：逐channel的对图像进行标准化

- `output = (input - mean) / std`

## 4.1 数据增强

数据增强又称为数据增广，数据扩增，它是对训练集进行变换，使训练集更丰富，从而让模型更具泛化能力

## 4.2 裁剪

`transforms.CenterCrop()`

- 功能：从图像中心裁剪图片
- size：所需裁剪图片尺寸  

---

```py
transforms.RandomCrop(size,
                      padding=None,
                      pad_if_needed=False,
                      fill=0,
                      padding_mode='constant')
```

功能：从图片中随机位置裁剪出尺寸为 size 的图片

- `size`：所需裁剪图片尺寸
- `padding`：设置填充大小
  - 当为 `a` 时，上下左右均填充 a 个像素
  - 当为 `(a, b)` 时，上下填充 b 个像素，左右填充 a 个像素
  - 当为 `(a, b, c, d)` 时，左，上，右，下分别填充 a, b, c, d
- `pad_if_need`：若图像小于设定 `size`，则填充
- `padding_mode`：填充模式，有4种模式
  - `constant`：像素值由 `fill` 设定
  - `edge`：像素值由图像边缘像素决定
  - `reflect`：镜像填充，最后一个像素不镜像， eg： `[1,2,3,4] → [3,2,1,2,3,4,3,2]`
  - `symmetric`：镜像填充，最后一个像素镜像， eg： `[1,2,3,4] → [2,1,1,2,3,4,4,3]`
- `fill`：`constant` 时，设置填充的像素值  

---

```python
RandomResizedCrop(size,
                  scale=(0.08, 1.0),
                  ratio=(3/4, 4/3),
                  interpolation)
```

功能：随机大小、长宽比裁剪图片

- `size`：所需裁剪图片尺寸
- `scale`：随机裁剪面积比例, 默认 `(0.08, 1)`
- `ratio`：随机长宽比，默认 `(3/4, 4/3)`
- `interpolation`：插值方法
  - `PIL.Image.NEAREST`
  - `PIL.Image.BILINEAR`
  - `PIL.Image.BICUBIC`

```py
transforms.FiveCrop(size)
transforms.TenCrop(size,
                   vertical_flip=False)
```

功能：在图像的上下左右以及中心裁剪出尺寸为 `size` 的 5 张图片，`TenCrop` 对这 5 张图片进行水平或者垂直镜像获得 10 张图片

- `size`：所需裁剪图片尺寸
- `vertical_flip`：是否垂直翻转  

## 4.3 翻转、旋转  

```py
RandomHorizontalFlip(p=0.5)
RandomVerticalFlip(p=0.5)
```

功能：依概率水平（左右）或垂直（上下）翻转图片

- p：翻转概率

---

```py
RandomRotation(degrees,
               resample=False,
               expand=False,
               center=None)
```

功能：随机旋转图片

- `degrees`：旋转角度
  - 当为a时，在 `(-a, a)` 之间选择旋转角度
  - 当为 `(a, b)` 时，在 `(a, b)` 之间选择旋转角度
- `resample`：重采样方法
- `expand`：是否扩大图片，以保持原图信息
- `center`：旋转点设置，默认中心旋转  

## 4.4 图像变换

### 4.4.1 边缘填充

```py
transforms.Pad(padding,
               fill=0,
               padding_mode='constant')
```

功能：对图片边缘进行填充

- `padding`：设置填充大小
  - 当为 a 时，上下左右均填充 a 个像素
  - 当为 `(a, b)` 时，上下填充b个像素，左右填充 a 个像素
  - 当为 `(a, b, c, d)` 时，左，上，右，下分别填充 a, b, c, d
- `padding_mode`：填充模式，有 4 种模式， constant、 edge、 reflect 和 symmetric
- `fill`： `constant` 时，设置填充的像素值，`(R, G, B)` or (Gray)  

### 4.4.2 图像调整

```py
transforms.ColorJitter(brightness=0,
                       contrast=0,
                       saturation=0,
                       hue=0)
```

功能：调整亮度、对比度、饱和度和色相

- `brightness`：亮度调整因子
  - 当为 a 时，从 `[max(0, 1-a), 1+a]` 中随机选择
  - 当为 `(a, b)` 时，从 `[a, b]` 中
- `contrast`：对比度参数，同 brightness
- `saturation`：饱和度参数，同 brightness
- `hue`：色相参数
  - 当为 $a$ 时，从 $[-a, a]$ 中选择参数，注： $0\le a \le 0.5$
  - 当为 $(a, b)$ 时，从 $[a, b]$ 中选择参数，注： $-0.5 \le a \le b \le 0.5$

---

### 4.4.3 灰度图转换

```py
RandomGrayscale(num_output_channels, p=0.1)
Grayscale(num_output_channels)
```

功能：依概率将图片转换为灰度图

- `num_ouput_channels`：输出通道数
  - 只能设 1 或 3
- `p`：概率值，图像被转换为灰度图的概率  

---

### 4.4.4 仿射变换

```py
RandomAffine(degrees,
             translate=None,
             scale=None,
             shear=None,
             resample=False,
             fillcolor=0)
```

功能：对图像进行仿射变换，仿射变换是二维的线性变换，由五种基本原子变换构成，分别是旋转、 平移、 缩放、 错切和翻转

- `degrees`：旋转角度设置
- `translate`：平移区间设置，如(a, b)，a 设置宽（width）， b 设置高(height)
  - 图像在宽维度平移的区间为 $\rm -img\_width \times a < dx < img\_width \times a$
- `scale`：缩放比例（以面积为单位）
- `fill_color`：填充颜色设置  
- `shear`：错切角度设置，有水平错切和垂直错切
  - 若为a，则仅在x轴错切，错切角度在(-a, a)之间
  - 若为(a， b)，则a设置x轴角度， b设置y的角度
  - 若为(a, b, c, d)，则a, b设置x轴角度， c, d设置y轴角度
- `resample`：重采样方式，有NEAREST 、 BILINEAR、 BICUBIC

### 4.4.5 随机遮挡

功能：对图像进行随机遮挡

```py
RandomErasing(p=0.5,
              scale=(0.02, 0.33),
              ratio=(0.3, 3.3),
              value=0,
              inplace=False)  
```

- `p`：概率值，执行该操作的概率
- `scale`：遮挡区域的面积
- `ratio`：遮挡区域长宽比
- `value`：设置遮挡区域的像素值， (R, G, B) or (Gray)

函数是对张量进行操作，之前的函数都是对 image 操作。所以在这个位置，首先需要转为张量 `image.ToTensor()`

### 4.4.6 Lambda

```py
transforms.Lambda(lambd)
```

功能：用户自定义 lambda 方法

- `lambd`： lambda 匿名函数

- `lambda [arg1 [,arg2, ... , argn]] : expression  `

- ```py
  transforms.TenCrop(200, vertical_flip=True),
  transforms.Lambda(lambda crops: torch.stack([transforms.Totensor()(crop) for crop in crops])),
  ```

## 4.5 选择

```py
transforms.RandomChoice([transforms1, transforms2, transforms3])
```

功能：从一系列transforms方法中随机挑选一个

```py
transforms.RandomApply([transforms1, transforms2, transforms3], p=0.5)
```

功能：依据概率执行一组transforms操作

```py
transforms.RandomOrder([transforms1, transforms2, transforms3])
```


功能：对一组transforms操作打乱顺序  

## 4.6 自定义

```py
class Compose(object):
    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img
```

要素

1. 仅接收一个参数，返回一个参数
2. 注意上下游的输出与输入

通过类实现多参数传入

```py
class YourTransforms(object):
    def __init__(self, ...):
    ...
    def __call__(self, img):
    ...
    	return img  
```

---

### 椒盐噪声

- 椒盐噪声又称为脉冲噪声，是一种随机出现的白点或者黑点, 白点称为盐噪声，黑色为椒噪声
- 信噪比（Signal-Noise Rate, SNR）是衡量噪声的比例，图像中为图像像素的占比  

```py
class AddPepperNoise(object):
    def __init__(self, snr, p):
        self.snr = snr # 信噪比
        self.p = p # 依概率增加噪声
    def __call__(self, img):
        # 添加椒盐噪声具体实现过程
    	return img
```

添加椒盐噪声具体实现过程


```py
class Compose(object):
    def __call__(self, img):
        for t in self.transforms:
        	img = t(img)
        return img
```

## 4.7 总结

### 4.7.1 裁剪

1. `transforms.CenterCrop`
2. `transforms.RandomCrop`
3. `transforms.RandomResizedCrop`
4. `transforms.FiveCrop`
5. `transforms.TenCrop`

### 4.7.2 翻转和旋转

1. `transforms.RandomHorizontalFlip`
2. `transforms.RandomVerticalFlip`
3. `transforms.RandomRotation`

### 4.7.3 图像变换

1. `transforms.Pad`
2. `transforms.ColorJitter`
3. `transforms.Grayscale`
4. `transforms.RandomGrayscale`
5. `transforms.RandomAffine`
6. `transforms.LinearTransformation`
7. `transforms.RandomErasing`
8. `transforms.Lambda`
9. `transforms.Resize`
10. `transforms.Totensor`
11. `transforms.Normalize`

### 4.7.4 Random

- `transforms.RandomChoice`
- `transforms.RandomApply`
- `transforms.RandomOrder  `

### 4.7.5 `Compose`

```py
inference_transform = transforms.Compose([
  transforms.Resize(256),
  transforms.CenterCrop((224, 224)),
  transforms.ToTensor(),
  transforms.Normalize(norm_mean, norm_std),
])
```

具体是对图像进行各种转换操作，并用函数 `Compose` 将这些转换操作组合起来

# 五、Net

## 5.1 模型创建与nn.Module

nn.Module 模型创建步骤

- 模型创建
  - 构建网络层：卷积层，池化层， 激活函数层等
  - 拼接网络层：LeNet， AlexNet， ResNet等
- 权值初始化：Xavier， Kaiming，均匀分布，正态分布等

---

模型构建两要素

- 构建子模块 `__init__()`
- 拼接子模块 `forward()`


---

`torch.nn`

- `nn.Parameter`：张量子类，表示可学习参数，如weight, bias
- `nn.Module`：所有网络层基类，管理网络属性
- `nn.functional`：函数具体实现，如卷积，池化，激活函数等
- `nn.init`：参数初始化方法

---

`parameters`：存储管理 `nn.Parameter` 类

`modules`：存储管理 `nn.Module` 类

`buffers`：存储管理缓冲属性，如 BN 层中的 `running_mean`

`***_hooks`：存储管理钩子函数  

```py
self._parameters = OrderedDict()
self._buffers = OrderedDict()
self._backward_hooks = OrderedDict()
self._forward_hooks = OrderedDict()
self._forward_pre_hooks = OrderedDict()
self._state_dict_hooks = OrderedDict()
self._load_state_dict_pre_hooks = OrderedDict()
self._modules = OrderedDict()
```

---

`nn.Module` 总结

- 一个 `module` 可以包含多个子 `module`
- 一个 `module` 相当于一个运算，必须实现 `forward()` 函数
- 每个 `module` 都有 8 个字典管理它的属性  

## 5.2 模型容器与AlexNet构建  

### 5.2.1 模型容器（Containers）

#### 5.2.1.1 Containers

- nn.Sequetial 按顺序包装多个网络层
- nn.ModuleDict 像python的dict一样包装多个网络层
- nn.ModuleList 像python的list一样包装多个网络层  

#### 5.2.1.2 Sequential

nn.Sequential 是 nn.module 的容器，用于==按顺序==包装一组网络层

- 顺序性：各网络层之间严格按照顺序构建
- 自带 `forward()`：自带的 `forward` 里，通过 `for` 循环依次执行前向传播运算

```py
class LeNetSequentialOrderDict(nn.Module):
    def __init__(self, classes):
        super(LeNetSequentialOrderDict, self).__init__()

        # 每一层可以命名，方便索引
        self.features = nn.Sequential(OrderedDict({
            'conv1': nn.Conv2d(3, 6, 5),
            'relu1': nn.ReLU(inplace=True),
            'pool1': nn.MaxPool2d(kernel_size=2, stride=2),

            'conv2': nn.Conv2d(6, 16, 5),
            'relu2': nn.ReLU(inplace=True),
            'pool2': nn.MaxPool2d(kernel_size=2, stride=2),
        }))

        self.classifier = nn.Sequential(OrderedDict({
            'fc1': nn.Linear(16*5*5, 120),
            'relu3': nn.ReLU(),

            'fc2': nn.Linear(120, 84),
            'relu4': nn.ReLU(inplace=True),

            'fc3': nn.Linear(84, classes),
        }))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x
```

#### 5.2.1.3 ModuleList

nn.ModuleList是 nn.module的容器，用于包装一组网络层，以==迭代==方式调用网络层

主要方法：

- `append()`： 在 ModuleList 后面<u>添加</u>网络层
- `extend()`：<u>拼接</u>两个 ModuleList
- `insert()`： 指定在 ModuleList 中位置<u>插入</u>网络层  

```py
class ModuleList(nn.Module):
    def __init__(self):
        super(ModuleList, self).__init__()
        # 容易构造出 20 层全连接层
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(20)]) 

    def forward(self, x):
        for i, linear in enumerate(self.linears):
            x = linear(x)
        return x
```

#### 5.2.1.4 ModuleDict

nn.ModuleDict 是 nn.module的容器，用于包装一组网络层，以==索引==方式调用网络层

主要方法：

- `clear()`： 清空ModuleDict
- `items()`： 返回可迭代的键值对(key-value pairs)
- `keys()`： 返回字典的键(key)
- `values()`： 返回字典的值(value)
- `pop()`： 返回一对键值，并从字典中删除

```py

class ModuleDict(nn.Module):
    def __init__(self):
        super(ModuleDict, self).__init__()
        self.choices = nn.ModuleDict({
            'conv': nn.Conv2d(10, 10, 3),
            'pool': nn.MaxPool2d(3)
        })

        self.activations = nn.ModuleDict({
            'relu': nn.ReLU(),
            'prelu': nn.PReLU()
        })

    def forward(self, x, choice, act):
        x = self.choices[choice](x)
        x = self.activations[act](x)
        return x

net = ModuleDict()
fake_img = torch.randn((4, 10, 32, 32))
output = net(fake_img, 'conv', 'relu')
print(output)
```

#### 5.2.1.5 总结

- nn.Sequential： 顺序性，各网络层之间严格按顺序执行，常用于block构建
- nn.ModuleList： 迭代性，常用于大量重复网构建，通过for循环实现重复构建
- nn.ModuleDict： 索引性，常用于可选择的网络层

### 5.1.2 AlexNet构建

AlexNet： 2012年以高出第二名10多个百分点的准确率获得ImageNet分类任务冠军，开创了卷积神经网络的新时代

AlexNet特点如下：

1. 采用ReLU：替换饱和激活函数，减轻梯度消失
2. 采用LRN(Local Response Normalization)：对数据归一化，减轻梯度消失
3. Dropout：提高全连接层的鲁棒性，增加网络的泛化能力
4. Data Augmentation： TenCrop，色彩修改  

<img src="https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20221012131523490.png" alt="image-20221012131523490" style="zoom:80%;" />

## 5.3 Convolution

### 5.3.1 1d/2d/3d Convolution

卷积运算：卷积核在输入信号（图像）上滑动，相应位置上进行乘加

卷积核：又称为滤波器，过滤器，可认为是某种模式，某种特征。

卷积过程类似于用一个模版去图像上寻找与它相似的区域，与卷积核模式越相似，激活值越高，从而实现特征提取  

卷积维度： 一般情况下，卷积核在几个维度上滑动，就是几维卷积  

### 5.3.2 nn.Conv2d

功能：对多个二维信号进行二维卷积

主要参数：

- in_channels：输入通道数
- out_channels：输出通道数，等价于卷积核个数
- kernel_size：卷积核尺寸
- stride：步长
- padding ：填充个数
- dilation：空洞卷积大小（一般用于提升感受野，卷积核扫描时的空洞大小）
- groups：分组卷积设置（模型轻量化）
- bias：偏置

```py
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
          dilation=1, groups=1, bias=True, padding_mode='zeros')
```

  ![image-20221012175925951](https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20221012175925951.png)

---

尺寸计算

- 简化版 $\rm\Large out_{size} = \frac{In_{size}-kernel_{size}}{stride}+1$
- 完整版

$$
\rm
H_{out}={\Large \lfloor}\frac{H_{in}+2\times padding[0]\times(kernel_size[0]-1)-1}{stride[0]}+1{\Large\rfloor}
$$



### 5.3.3 转置卷积 ConvTranspose2d

转置卷积又称为反卷积(Deconvolution)和部分跨越卷积(Fractionally-strided Convolution) ，用于对图像进行上采样(UpSample)

为什么称为转置卷积？

正常卷积：假设图像尺寸为 $4\times 4$，卷积核为 $3\times3$，`padding=0`，`stride=1`

- 图像：$I_{16\times1}$
- 卷积核： $k_{4\times16}$ 变成 16 的原因是补 0
- 输出： $O_{4\times1}=K_{4\times16}\times I_{16\times 1}$

转置卷积：假设图像尺寸为 $2\times 2$，卷积核为 $3\times3$，`padding=0`，`stride=1`

- 图像：$I_{4\times1}$
- 卷积核：$k_{16\times 4}$
- 输出：$O_{16\times1}=K_{16\times4}\times I_{4\times 1}$

---

功能：转置卷积实现上采样

```py
nn.ConvTranspose2d(in_channels,
                   out_channels,
                   kernel_size,
                   stride=1,
                   padding=0,
                   output_padding=0,
                   groups=1,
                   bias=True,
                   dilation=1,
                   padding_mode='zeros')
```

主要参数：

- in_channels：输入通道数
- out_channels：输出通道数
- kernel_size：卷积核尺寸
- stride：步长
- padding ：填充个数  
- dilation：空洞卷积大小
- groups：分组卷积设置
- bias：偏置  

尺寸计算：

- 简化版
  - $\Large \rm Out_{size} = (In_{size}-1)\times stride + kernel_{size}$
  - $\Large\rm Out_{size}=\frac{In_{size}-kernel_{size}}{stride}+1$
- 完整版

$$
\rm H_{out}=(H_{in}-1)\times stride[0]+dilation[0]\times(kernel_size[0]-1)+output[0]+1
$$

会产生[棋盘效应](https://distill.pub/2016/deconv-checkerboard/)

## 5.4 Pooling

池化运算：对信号进行 “收集”并 “总结”，类似水池收集水资源，因而得名池化层

“收集”：多变少 “总结”：最大值/平均值

### 5.4.1 `MaxPool2d`

功能：对二维信号（图像）进行最大值

```py
nn.MaxPool2d(kernel_size, stride=None,
             padding=0, dilation=1,
             return_indices=False,
             ceil_mode=False)
```

主要参数：

- kernel_size：池化核尺寸
- stride：步长
- padding ：填充个数
- dilation：池化核间隔大小
- ceil_mode：尺寸向上取整
- return_indices：记录池化像素索  

---

### 5.4.2 `AvgPool2d`

功能：对二维信号（图像）进行平均值池化

```py
nn.AvgPool2d(kernel_size,
             stride=None,
             padding=0,
             ceil_mode=False,
             count_include_pad=True,
             divisor_override=None)
```

主要参数

- kernel_size：池化核尺寸
- stride：步长
- padding：填充个数
- ceil_mode：尺寸向上取整
- count_include_pad：填充值用于计算
- divisor_override：除法因子  

### 5.4.3 MaxUnpool2d

功能：对二维信号（图像）进行最大值池化上采样

```py
nn.MaxUnpool2d(kernel_size,
               stride=None,
               padding=0)
forward(self, input, indices, output_size=None)
```

主要参数

- kernel_size：池化核尺寸
- stride：步长
- padding：填充个数  

## 5.5 Linear

线性层又称全连接层，其每个神经元与上一层所有神经元相连

实现对前一层的线性组合， 线性变换
$$
\begin{align}
\rm
Input &= [1,2,3]\quad &\text{shape}=(1,3)\\
W_0&=\left[ {\begin{array}{*{20}{c}}
1&2&3&4\\
1&2&3&4\\
1&2&3&4\\
1&2&3&4
\end{array}} \right]
\quad &\text{shape}=(3,4)\\
\text{Hidden}&=\text{Input}\times W_0=[6,12,18,24]
\quad &\text{shape}=(1,4)
\end{align}
$$

```py
nn.Linear(in_features, out_features, bias=True)
```

功能：对一维信号（向量）进行线性组合

主要参数：

- in_features：输入结点数
- out_features：输出结点数
- bias ：是否需要偏置

计算公式：$y=xW^T+bias$

## 5.6 Activation

激活函数对特征进行非线性变换，赋予多层神经网络具有深度的意义
$$
\begin{align}
H_1&=X\times W_1\\
H_2&=H_1\times W_2\\
Output &= H_2\times W_3\\
&=H_1\times W_2\times W_3\\
&=X\times(W_1\times W_2\times W_3)\\
&=X\times W
\end{align}
$$

### 5.6.1 Sigmoid

计算公式：$y=(1+e^{-x})^{-1}$

梯度公式：$y'=y\times(1-y)$

特性：

- 输出值在 $(0,1)$，符合概率
- 导数范围是 $[0, 0.25]$，易导致梯度消失
- 输出为非 0 均值，破坏数据分布

![](https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20221012215355996.png)

### 5.6.2 tanh

计算公式
$$
y=\frac{\sin x}{\cos x}=\frac{e^x-e^{-x}}{e^{x}+e^{-x}}=\frac{2}{1+e^{-2x}}+1
$$
梯度公式：$y'=1-y^2$

特性

- 输出值在(-1,1)，数据符合0均值
- 导数范围是(0, 1)，易导致梯度消失  

![image-20221012215820848](https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20221012215820848.png)

### 5.6.3 ReLU

计算公式：$y=\max(0,x)$

梯度公式
$$
y = 
\left\{ 
{\begin{array}{*{20}{c}}
1&{x > 0}\\
\text{undefined} &{x = 0}\\
0&{x < 0}
\end{array}} \right.
$$
特性：

- 输出值均为正数，负半轴导致死神经元（网络会变得更加稀疏）
- 导数是1，缓解梯度消失，但易引发梯度爆炸  

![image-20221012215826056](https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20221012215826056.png)



nn.LeakyReLU

- negative_slope：负半轴斜率

nn.PReLU

- init：可学习斜率

nn.RReLU

- lower：均匀分布下限
- upper：均匀分布上限  

![image-20221012215834010](https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20221012215834010.png)

# 六、权值初始化

## 6.1 梯度消失与爆炸

$$
\begin{align}
H_2&=H_1\times W_2\\
\bigtriangleup W_2&=\frac{\rm \partial Loss}{\partial W_2}=\frac{\rm\partial Loss}{\partial\rm out}\times\frac{\rm\partial out}{\partial H_2}\times\frac{\rm \partial H_2}{\partial W_2}\\
&=\frac{\rm Loss}{\partial\rm out}\times \frac{\rm out}{\partial H_2}\times H_1
\end{align}
$$

梯度消失：$H_1\to0\Rightarrow \bigtriangleup W_2 \to 0$

梯度爆炸：$H_1\to\infty\Rightarrow \bigtriangleup W_2 \to \infty$

![image-20221013174044637](https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20221013174044637.png)

## 6.2 Xavier 初始化

方差一致性：保持数据尺度维持在恰当范围，通常方差为 1

激活函数：饱和函数，如 Sigmoid，Tanh
$$
\begin{align}
n_i \times D(W)&=1\\
n_{i+1}\times D(W)&=1
\end{align}
$$
$n_i$ 输入层的神经元个数

$n_{i+1}$ 输出层的神经元个数

```py
a = np.sqrt(6 / (self.neural_num + self.neural_num))

tanh_gain = nn.init.calculate_gain('tanh')
a *= tanh_gain

nn.init.uniform_(m.weight.data, -a, a)

nn.init.xavier_uniform_(m.weight.data, gain=tanh_gain)
```

## 6.3 Kaiming 初始化

方差一致性：保持数据尺度维持在恰当范围，通常方差为1

激活函数： ReLU 及其变种  
$$
\begin{align}
D(W)=\frac{2}{n_i}\\
D(W)=\frac{2}{(1+a^2)\times n_i}\\
\operatorname{std}(W)=\sqrt{\frac{2}{(1+a^2)\times n_i}}
\end{align}
$$

```py
nn.init.normal_(m.weight.data, std=np.sqrt(2 / self.neural_num))
nn.init.kaiming_normal_(m.weight.data)
```

## 6.3 常用初始化方法

十种初始化方法

1. Xavier 均匀分布
2. Xavier 正态分布
3. Kaiming均匀分布
4. Kaiming正态分布
5. 均匀分布
6. 正态分布
7. 常数分布
8. 正交矩阵初始化
9. 单位矩阵初始化
10. 稀疏矩阵初始化  

```py
nn.init.calculate_gain(nonlinearity, param=None)
```

主要功能：计算激活函数的方差变化尺度

主要参数

- nonlinearity：激活函数名称
- param：激活函数的参数，如 Leaky ReLU 的 negative _slop  

# 七、损失函数

损失函数：衡量模型输出与真实标签的差异

损失函数(Loss Function)：$Loss = f(\hat y,y)$

代价函数(Co st Function)：$Cost = \frac{1}{N}\sum\limits_{i}^{N}f(\hat y_i,y_i)$

目标函数(Objective Function)：$\rm Obj = Cost + Regularization$

## 7.1 CrossEntropyLoss

功能：`nn.LogSoftmax()` 与 `nn.NLLLoss()` 结合，进行交叉熵计算

```py
nn.CrossEntropyLoss(weight=None,
                    ignore_index=-100,
                    reduction=‘mean’‘)
```

主要参数：

- weight：各类别的loss设置权值
- ignore_index：忽略某个类别
- reduction ：计算模式，可为none/sum/mean
  - none- 逐个元素计算
  - sum- 所有元素求和，返回标量
  - mean- 加权平均，返回标量  

$$
\begin{align}
H(P,Q)&=-\sum\limits_{i=1}^N P(x_i)\log Q(x_i)\\
loss(x,class)&=-\log(\frac{\exp(x[class])}{\sum_j\exp(e[j])})=-x[class]+\log(\sum\limits_j \exp(x[j]))\\
loss(x,class)&=weight[class]\left(-x[class]+\log(\sum\limits_j \exp(x[j]))\right)

\end{align}
$$

## 7.2 NLLLoss

功能： 实现负对数似然函数中的负号功能

主要参数：

- weight：各类别的loss设置权值
- ignore _index：忽略某个类别
- reduction：计算模式，可为none/sum/mean
  - none-逐个元素计算
  - sum-所有元素求和，返回标量
  - mean-加权平均，返回标量  

```py
nn.NLLLoss(weight=None,
           ignore_index=-100,
           reduction='mean')
```

$$
l(x,y)=L=\{l_1,...,l_N\}^T,\quad l_n=-w_{y_n}x_{n,y_n}
$$

## 7.3 BCELoss

功能： 二分类交叉熵

注意事项：输入值取值在[0,1]

主要参数：

- weight：各类别的loss设置权值
- ignore_index：忽略某个类别
- reduction ：计算模式，可为none/sum /mean
  - none-逐个元素计算
  - sum-所有元素求和，返回标量
  - mean-加权平均，返回标量  

```py
nn.BCELoss(weight=None,
           reduction='mean’)
```

$$
l_n=-w_n[y_n\cdot\log x_n + (1-y_n)\cdot\log(1-x_n)]
$$

## 7.4 BCEWithLogitsLoss

功能：结合Sigmoid与二分类交叉熵

注意事项：网络最后不加sigmoid函数

主要参数：

- pos _weigh t ：正样本的权值
- weight：各类别的loss设置权值
- ignore_index：忽略某个类别
- reduction：计算模式，可为none/sum/mean
  - none-逐个元素计算
  - sum-所有元素求和，返回标量
  - mean-加权平均，返回标量

```py
nn.BCEWithLogitsLoss(weight=None,
                     reduction='mean',
                     pos_weight=None)
```

$$
l_n=-w_n[y_n\cdot\log\sigma(x_n)+(1-y_n)\cdot\log(1-\sigma(x_n))]
$$

## 7.5 L1 Loss

- 功能： 计算inputs与target之差的绝对值

主要参数：

- reduction ：计算模式，可为none/sum/mean
  - none - 逐个元素计算
  - sum - 所有元素求和，返回标量
  - mean - 加权平均，返回标量

```py
nn.L1Loss(reduction='mean’)
```

$$
l_n=|x_n-y_n|
$$

## 7.6 MSE Loss

功能： 计算input s与target之差的平方

reduction 参数同上

```py
nn.MSELoss(reduction='mean’)
```

$$
l_n=(x_n-y_n)^2
$$

## 7.7 SmoothL1Loss

功能： 平滑的 L1 Loss

主要参数

- reduction ：计算模式，可为none/sum/mean
  - none- 逐个元素计算
  - sum- 所有元素求和，返回标量
  - mean- 加权平均，返回标量  

```py
nn.SmoothL1Loss(reduction='mean’)
```

$$
\operatorname{loss}(x,y)=\frac{1}{n}\sum\limits_i z_i\\

{z_i} = \left\{ {\begin{array}{*{20}{c}}
{0.5{{\left( {{x_i} - {y_i}} \right)}^2},}&{{\rm{if}}\;\left| {{x_i} - {y_i}} \right| < 1}\\
{\left| {{x_i} - {y_i}} \right|-0.5,}&{{\rm{otherwise}}}
\end{array}} \right.
$$

<img src="https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20221014135328353.png" alt="image-20221014135328353" style="zoom:80%;" />

## 7.8 PoissonNLLLoss

功能：泊松分布的负对数似然损失函数

```py
nn.PoissonNLLLoss(log_input=True, full=False,
                  eps=1e-08,
                  reduction='mean')
```

主要参数：

- `log_input`：输入是否为对数形式，决定计算公式
- full ：计算所有 loss，默认为False
- eps ：修正项，避免 `log(input)` 为 `nan`

```py
log_input = True
loss(input, target) = exp(input) - target * input
log_input = False
loss(input, target) = input - target * log(input+eps)
```

## 7.9 KLDivLoss

功能：计算 KLD（ divergence）， KL散度，相对熵

注意事项：需提前将输入计算 log-probabilities，如通过 `nn.logsoftmax()`

```py
nn.KLDivLoss(reduction='mean')
```

主要参数：

- reduction ： none/sum/mean/batchmean
  - batchmean- batchsize维度求平均值
  - none- 逐个元素计算
  - sum- 所有元素求和，返回标量
  - mean- 加权平均，返回标量  

$$
\begin{align}
D_{KL}(P||Q)=E_{x\sim p}\left[\log\frac{P(x)}{Q(x)}\right]
&=E_{x\sim p}\left[\log P(x)-\log Q(x)\right]\\
&=\sum\limits_{i=1}^{N}P(x_i)\left(\log P(x_i)-\log Q(x_i)\right)\\
l_n=y_n\cdot(\log y_n-x_n)
\end{align}
$$

## 7.10 MarginRankingLoss

功能： 计算两个向量之间的相似度，用于排序任务

特别说明：该方法计算两组数据之间的差异，返回一个 $n\times n$ 的 loss 矩阵

```py
nn.MarginRankingLoss(margin=0.0, reduction='mean')
```

主要参数：

- margin ：边界值， x1 与 x2 之间的差异值
- reduction ：计算模式，可为 none/sum/mean
- $y = 1$ 时， 希望 $x_1$ 比 $x_2$ 大，当 $x_1>x_2$ 时，不产生 loss
- $y = -1$ 时，希望 $x_2$ 比 $x_1$ 大，当 $x_2>x_1$ 时，不产生 loss  

$$
\operatorname{loss}(x,y)=\max\left(0,-y\times(x_1-x_w)+\text{margin}\right)
$$

## 7.11 MultiLabelMarginLoss

功能： 多标签边界损失函数（多标签：一张图片可能对应多个类别，有云，有草地，有。。。）

举例：四分类任务，样本x属于0类和3类，

标签： [0, 3, -1, -1] , 不是 [1, 0, 0, 1]

主要参数：

- reduction ：计算模式，可为none/sum/mean

```py
nn.MultiLabelMarginLoss(reduction='mean')
```

$$
\operatorname{loss}(x, y)=\sum_{i j} \frac{\max (0,1-(x[y[j]]-x[i]))}{\mathrm{x} . \operatorname{size}(0)}
$$
where $i==0$ to $x.\operatorname{size}(0)$, $j==0$ to $y.\operatorname{size}(0), y[j] \geq 0$, and $i \neq y[j]$ for all $i$ and $j$.

$x[y[j]]-x[i]$ 标签所在神经元 减去 非标签所在神经元

## 7.12 SoftMarginLoss

功能： 计算二分类的logistic损失

主要参数：

- reduction ：计算模式，可为 none/sum/mean  

```py
nn.SoftMarginLoss(reduction='mean')
```

$$
\operatorname{loss}(x,y)=\sum\limits_{i}\frac{\log\left(1-\exp(-y[i]\times x[i])\right)}{x.\text{nelement}()}
$$

## 7.13 MultiLabelSoftMarginLoss

功能： SoftMarginLoss多标签版本

主要参数：

- weight：各类别的loss设置权值
- reduction：计算模式，可为none/sum/mean  

```py
nn.MultiLabelSoftMarginLoss(weight=None, reduction='mean')
```

$$
\operatorname{loss}(x, y)=-\frac{1}{C}  \sum_i y[i] * {\log \left((1+\exp (-x[i]))^{-1}\right)}+{(1-y[i])} * \log \left(\frac{\exp (-x[i])}{1+\exp (-x[i])}\right)
$$

## 7.14 MultiMarginLoss

功能：计算多分类的折页损失

```py
nn.MultiMarginLoss(p=1, margin=1.0, reduce=None, reduction='mean')
```

主要参数

- p：可选 1 或 2
- weight：各类别的 loss 设置权值
- margin：边界值
- reduction：计算模式，可为none/sum/mean

$$
\operatorname{loss}(x, y)=\frac{\left.\sum_i \max (0, \operatorname{margin}-x[y]+x[i])\right)^p}{\mathrm{x} \cdot \operatorname{size}(0)}
$$

where $x\in\{0,...,x.\text{size}(0)-1\}$，$y\in\{0,...,y.\text{size}(0)-1\}$，$0\le y[i]\le x.\text{size}(0)-1$，and $i\neq y[j]$ for all $i$ and $j$

## 7.15 TripletMarginLoss

功能：计算三元组损失，人脸验证中常用

```py
nn.TripletMarginLoss(margin=1.0, p=2.0, eps=1e-06,
                     swap=False, reduction='mean')
```

主要参数

- p ：范数的阶，默认为2
- margin ：边界值
- reduction ：计算模式，可为none/sum/mean  

$$
\begin{gathered}
L(a, p, n)=\max \left\{d\left(a_i, p_i\right)-d\left(a_i, n_i\right)+\operatorname{margin}, 0\right\} \\
d\left(x_i, y_i\right)=\left\|\mathbf{x}_i-\mathbf{y}_i\right\|_p
\end{gathered}
$$

![image-20221014145520495](https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20221014145520495.png)

## 7.16 HingeEmbeddingLoss

功能：计算两个输入的相似性，常用于非线性embedding和半监督学习

特别注意：输入x应为两个输入之差的绝对值

主要参数：

- margin：边界值
- reduction：计算模式，可为none/sum/mean  

```py
nn.HingeEmbeddingLoss(margin=1.0, reduction='mean’)
```

$$
l_n= \begin{cases}x_n, & \text { if } y_n=1 \\ \max \left\{0, \Delta-x_n\right\}, & \text { if } y_n=-1\end{cases}
$$

## 7.17 CosineEmbeddingLoss

功能： 采用余弦相似度计算两个输入的相似性

主要参数

- margin ：可取值[-1, 1] , 推荐为[0, 0.5]
- reduction ：计算模式，可为none/sum/mean  

```py
nn.CosineEmbeddingLoss(margin=0.0, reduction='mean')
```

$$
\begin{aligned}
&\operatorname{loss}(x, y)= \begin{cases}1-\cos \left(x_1, x_2\right), & \text { if } y=1 \\
\max \left(0, \cos \left(x_1, x_2\right)-\operatorname{margin}\right), & \text { if } y=-1\end{cases} \\
&\cos (\theta)=\frac{A \cdot B}{\|A \|\| B \|}=\frac{\sum_{i=1}^n A_i \times B_i}{\sqrt{\sum_{i=1}^n\left(A_i\right)^2} \times \sqrt{\sum_{i=1}^n\left(B_i\right)^2}}
\end{aligned}
$$

## 7.18 CTCLoss

功能： 计算 CTC 损失，解决时序类数据的分类

主要参数：

- blank： blank label
- zero_infinity：无穷大的值或梯度置0
- reduction：计算模式，可为none/sum/mean  

```py
torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=False)
```

# 八、优化器

pytorch的优化器： ==管理==并==更新==模型中可学习参数的值，使得模型输出更接近真实标签

导数：函数在指定坐标轴上的变化率

方向导数：指定方向上的变化率

梯度：一个向量，方向为方向导数取得最大值的方向  

## 8.1 基本属性

- `defaults`：优化器超参数
- `state`：参数的缓存，如 momentum 的缓存
- `params_groups`：管理的参数组
- `_step_count`：记录更新次数，学习率调整中使用  

```py
class Optimizer(object):
    def __init__(self, params, defaults):
        self.defaults = defaults
        self.state = defaultdict(dict)
        self.param_groups = []
        param_groups = [{'params': param_groups}]
```

## 8.2 基本方法

- `zero_grad()`：清空所管理参数的梯度
- `step()`：执行一步更新
- `add_param_group()`：添加参数组
- `state_dict()`：获取优化器当前状态信息字典
- `load_state_dict()`：加载状态信息字典 

pytorch 特性：张量梯度不自动清零

```py
class Optimizer(object):
    def __init__(self, params, defaults):
        self.defaults = defaults
        self.state = defaultdict(dict)
        self.param_groups = []

    def zero_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()
  
	def add_param_group(self, param_group):
        for group in self.param_groups:
            param_set.update(set(group['params’]))
                        
        self.param_groups.append(param_group)
                                       
	def state_dict(self):
        ...
		return {'state': packed_state,
                'param_groups': param_groups,}

	def load_state_dict(self, state_dict):
		pass
```

## 8.3 学习率

$$
\begin{align}
w_{i+1}&=w_i+g(w_i)\\
w_{i+1}&=w_i-LR\times g(w_i)

\end{align}
$$

学习率（ learning rate）控制更新的步伐

## 8.4 Momentum

动量，冲量：结合当前梯度与上一次更新信息，用于当前更新

梯度下降
$$
w_{i+1}=w_i-lr\times g(w_i)
$$


pytorch中更新公式
$$
\begin{align}
v_i&=m\times v_{i-1}+g(w_i)\\
w_{i+1}&=w_i-lr\times v_i
\end{align}
$$
$w_{i+1}$： 第 $i+1$ 次更新的参数

lr：学习率

$v_i$：更新量

$m$：momentum系数

$g(w_i)$：$w_i$ 的梯度 

## 8.5 优化器

```py
optim.SGD(params, lr=<object object>,
          momentum=0, dampening=0,
          weight_decay=0, nesterov=False)
```

主要参数

- params：管理的参数组
- lr：初始学习率
- momentum：动量系数，贝塔
- weight_decay： L2正则化系数
- nesterov：是否采用NAG  

---

1. optim.SGD：随机梯度下降法
2. optim.Adagrad：自适应学习率梯度下降法
3. optim.RMSprop： Adagrad的改进
4. optim.Adadelta： Adagrad的改进
5. optim.Adam： RMSprop结合Momentum
6. optim.Adamax： Adam增加学习率上限
7. optim.SparseAdam：稀疏版的Adam
8. optim.ASGD：随机平均梯度下降
9. optim.Rprop：弹性反向传播
10. optim.LBFGS： BFGS的改进  

---

# 九、学习率调整策略

```py
class _LRScheduler(object):
	def __init__(self, optimizer, last_epoch=-1):
        pass
    def get_lr(self):
        raise NotImplementedError
```


主要属性

- `optimizer`：关联的优化器
- `last_epoch`：记录 epoch 数
- `base_lrs`：记录初始学习率  

主要方法

- `step()`：更新下一个 epoch 的学习率（注意放在epoch的循环中）
- `get_lr()`：虚函数， 计算下一个 epoch 的学习率  

---

## 9.1 StepLR

功能：等间隔调整学习率

```py
lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)
```

主要参数

- `step_size`：调整间隔数（一个值，所以必须等间隔）
- `gamma`：调整系数
- 调整方式：$lr = lr \times gamma$

## 9.2 MultiStepLR

功能：按给定间隔调整学习率

主要参数：

- milestones：设定调整时刻数（一个列表，多个间隔）
- gamma：调整系数
- 调整方式：$lr = lr \times gamma$

```py
lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
```

## 9.3 ExponentialLR

功能：按指数衰减调整学习率

主要参数

- gamma：指数的底
- 调整方式：$\rm lr = lr \times gamma ^ {epoch}$

```py
lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)
```

## 9.4 CosineAnnealingLR

功能：余弦周期调整学习率

主要参数

- T_max：下降周期
- eta_min：学习率下限  

调整方式
$$
\eta_t = \eta_{\min}+\frac{1}{2}(\eta_{\max}-\eta_{\min})(1+\cos(\frac{T_{\rm cur}}{T_{\max}}\pi))
$$

```py
lr_scheduler.CosineAnnealingLR(optimizer,T_max, eta_min=0, last_epoch=-1)
```

## 9.5 ReduceLRonPlateau

功能：监控指标，当指标不再变化则调整

主要参数：

- `mode`：min/max 两种模式
- `factor`：调整系数
- `patience`：“耐心 ”，接受几次不变化
- `cooldown`：“冷却时间”，停止监控一段时间
- `verbose`：是否打印日志
- `min_lr`：学习率下限
- `eps`：学习率衰减最小值  

```py
lr_scheduler.ReduceLROnPlateau(optimizer,
                               mode='min', factor=0.1, patience=10,
                               verbose=False, threshold=0.0001,
                               threshold_mode='rel', cooldown=0, min_lr=0,
                               eps=1e-08)
```

## 9.6 LambdaLR

功能：自定义调整策略

主要参数

- `lr_lambda`：function or list 

```py
lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
```

用于设置不同参数组，有不同的学习率调整策略

## 9.7 小结

1. 有序调整：Step、MultiStep、Exponential 和 CosineAnnealing
2. 自适应调整：ReduceLROnPleateau
3. 自定义调整：Lambda  

学习率初始化

1. 设置较小数：0.01、0.001、0.0001
2. 搜索最大学习率：《Cyclical Learning Rates for Training Neural Networks》  

# 十、TensorBoard

## 10.1 SummaryWriter

功能：提供创建event file的高级接口

主要属性：

- log_dir： event file输出文件夹
- comment：不指定log_dir时，文件夹后缀
- filename_suffix： event file文件名后缀

```py
class SummaryWriter(object):
    def __init__(self, log_dir=None, comment='',
		purge_step=None, max_queue=10,
        flush_secs=120, filename_suffix=’’)
```

### 10.1.1 `add_scalar()`

功能：记录标量  

- tag：图像的标签名，图的唯一标识
- scalar_value：要记录的标量
- global_step：x轴

```py
add_scalar(tag, scalar_value, global_step=None, walltime=None)
```

### 10.1.2 `add_scalars()`

main_tag：该图的标签

tag_scalar_dict：key是变量的 tag，value 是变量的值  

```py
add_scalars(main_tag, tag_scalar_dict, global_step=None, walltime=None)
```

### 10.1.3 `add_histogram()`

功能：统计直方图与多分位数折线图

- tag：图像的标签名，图的唯一标识
- values：要统计的参数
- global_step：y轴
- bins：取直方图的bins  

```py
add_histogram(tag, values, global_step=None, bins='tensorflow', walltime=None)
```

### 10.1.4 `add_image()`

功能：记录图像

- `tag`：图像的标 签名，图的唯一标识
- `img_tensor`：图像数据，注意尺度
- `global_step`：x轴
- `dataformats`：数据形式， CHW， HWC， HW  

```py
add_image(tag, img_tensor, global_step=None, walltime=None, dataformats='CHW')
```

## 10.2 `make_grid`

```py
make_grid(tensor, nrow=8, padding=2,
          normalize=False, range=None, scale_each=False,
          pad_value=0)
```

功能：制作网格图像

- tensor：图像数据, BCHW形式（batch size, channel, height, width）
- nrow：行数（列数自动计算）
- padding：图像间距（像素单位）
- normalize：是否将像素值标准化
- range：标准化范围
- scale_each：是否单张图维度标准化
- pad_value：padding的像素值  

### 10.1.5 `add_graph()`

功能：可视化模型计算图

- model：模型，必须是 nn.Module
- input_to_model：输出给模型的数据
- verbose：是否打印计算图结构信息  

```py
add_graph(model, input_to_model=None, verbose=False)
```

一般是 pytorch 

## 10.3 torchsummary

功能：查看模型信息，便于调试

- `model`：pytorch模型
- `input_size`：模型输入size
- `batch_size`：batch size
- `device`：“cuda” or “cpu”  

```py
summary(model, input_size, batch_size=-1, device="cuda")
```

[github](https://github.com/sksq96/pytorch-summary)

# 十一、Hook函数与CAM算法

## 11.1 Hook Function

Hook函数机制：不改变主体，实现额外功能，像一个挂件，挂钩， hook

1. `torch.Tensor.register_hook(hook)`
2. `torch.nn.Module.register_forward_hook`
3. `torch.nn.Module.register_forward_pre_hook`
4. `torch.nn.Module.register_backward_hook`

### 11.1.1 Tensor.register_hook

功能：注册一个反向传播 hook 函数

Hook函数仅一个输入参数，为张量的梯度  

```py
hook(grad) -> Tensor or None
```

### 11.1.2 Module.register_forward_hook

功能：注册module的前向传播hook函数

参数：

- module：当前网络层
- input：当前网络层输入数据
- output：当前网络层输出数据  

```py
hook(module, input, output) -> None
```

### 11.1.3 Module.register_forward_pre_hook

功能：注册module前向传播前的hook函数

参数：

- module：当前网络层
- input：当前网络层输入数据  

```py
hook(module, input) -> None
```

### 11.1.4 Module.register_backward_hook

功能：注册module反向传播的hook函数

参数：

- module：当前网络层
- grad_input：当前网络层输入梯度数据
- grad_output：当前网络层输出梯度数据

```py
hook(module, grad_input, grad_output) -> Tensor or None
```

## 11.2 CAM and Grad-CAM

### 11.2.1 CAM

类激活图， class activation map

GAP：global average pooling

![image-20221018173033619](https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20221018173033619.png)

CAM： 《Learning Deep Features for Discriminative Localization》 

---

### 11.2.2 Grad-CAM

 CAM改进版，利用梯度作为特征图权重

![image-20221018173313965](https://raw.githubusercontent.com/plum-Yin/clould_picgo/main/pic/image-20221018173313965.png)

Grad-CAM： 《Grad-CAM:Visual Explanations from Deep Networks via Gradient-based Localization》  

# 十二、正则化

Regularization：减小==方差==的策略

误差可分解为：偏差，方差与噪声之和。即误差 = 偏差 + 方差 + 噪声之和

- ==偏差==度量了学习算法的期望预测与真实结果的偏离程度，即刻画了学习算法本身的拟合能力
- ==方差==度量了同样大小的训练集的变动所导致的学习性能的变化，即刻画了数据扰动所造成的影响
- ==噪声==则表达了在当前任务上任何学习算法所能达到的期望泛化误差的下界

## 12.1 weight decay

L2 Regularization = weight decay（权值衰减）
$$
\begin{align}
\text{Obj}&=\text{Loss}+\frac{\lambda}{2}\times \sum\limits_{i}^{N} w_i^2\\
w_{i+1}&=w_i-\frac{\partial \text{Obj}}{\partial w_i}=w_i-(\frac{\partial \text{Loss}}{\partial w_i}+\lambda\times w_i)\\
&=w_i(1-\lambda)-\frac{\partial \text{Loss}}{\partial w_i}
\end{align}
$$

## 12.2 Batch normalizaton

批：一批数据，通常为mini-batch

标准化： 0均值， 1方差

优点：

1. 可以用更大学习率，加速模型收敛
2. 可以不用精心设计权值初始化
3. 可以不用dropout或较小的dropout
4. 可以不用L2或者较小的weight decay
5. 可以不用LRN(local response normalization)

---

计算方式

<img src="https://s2.loli.net/2022/10/19/awcqzWl2QieBCbS.png" alt="image-20221020005712569" style="zoom:30%;" />

最后的 scale 和 shift 称为 Affine transform 增强 capacity

《Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift》

## 12.3 `_BatchNorm`

```py
__init__(self, num_features,
         eps=1e-5,
         momentum=0.1,
         affine=True,
         track_running_stats=True)
```

参数

- num_features：一个样本特征数量（最重要）
- eps：分母修正项
- momentum：指数加权平均估计当前 mean/var
- affine：是否需要 affine transform
- track_running_stats：是训练状态，还是测试状态

---

- nn.BatchNorm1d
- nn.BatchNorm2d
- nn.BatchNorm3d

$$
\begin{align}
\widehat x_i &\leftarrow \frac{x_i-\mu_{\mathcal{B}}}{\sqrt{\sigma^2_{\mathcal{B}}+\varepsilon}}\\

y_i &\leftarrow \gamma\widehat x_i +\beta=\text{BN}_{\gamma,\beta}(x_i)
\end{align}
$$

主要属性

- running_mean：均值
- running_var：方差
- weight：affine transform 中的 gamma
- bias：affine transform 中的 beta

训练：均值和方差采用指数加权平均计算

测试：当前统计值
$$
\rm
running\_mean = (1 - momentum) \times pre\_running\_mean + momentum \times mean\_t\\

\rm
running\_var = (1 - momentum) \times pre\_running\_var + momentum \times var\_t
$$

---

- nn.BatchNorm1d input= $B\times \text{特征数} \times 1d\text{ 特征}$
- nn.BatchNorm2d input= $B\times \text{特征数} \times 2d\text{ 特征}$
- nn.BatchNorm3d input= $B\times \text{特征数} \times 3d\text{ 特征}$

## 12.4 BN、LN、IN and GN

Why Normalization

- Internal Covariate Shift (ICS)：数据尺度/分布异常，导致训练困难

常见的Normalization
1. Batch Normalization（BN）
2. Layer Normalization（LN）
3. Instance Normalization（IN）
4. Group Normalization（GN）

---

### 12.4.1 Layer

Layer Normalization

起因： BN不适用于变长的网络，如RNN

思路： 逐层计算均值和方差

注意事项：

1. 不再有 running_mean 和 running_var
2. gamma 和 beta 为逐元素的

```py
nn.LayerNorm(normalized_shape,
             eps=1e-05,
             elementwise_affine=True)
```

主要参数：

- normalized_shape：该层特征形状
- eps：分母修正项
- elementwise_affine：是否需要 affine transform，是否包含学习参数

### 12.4.2 Instance

Instance Normalization

起因：BN在图像生成（Image Generation）中不适用

思路：逐 Instance（ channel）计算均值和方差

```py
nn.InstanceNorm2d(num_features,
                  eps=1e-05,
                  momentum=0.1,
                  affine=False,
                  track_running_stats=False)
```

主要参数

- num_features：一个样本特征数量（最重要）
- eps：分母修正项
- momentum：指数加权平均估计当前mean/var
- affine：是否需要affine transform
- track_running_stats：是训练状态，还是测试状态

### 12.4.3 Group

Group Normalization

起因：小batch样本中， BN估计的值不准

思路： 数据不够，通道来凑

注意事项：

1. 不再有running_mean和running_var
2. gamma 和 beta 为逐通道（ channel）的

应用场景：大模型（小batch size）任务

```py
nn.GroupNorm(num_groups,
             num_channels,
             eps=1e-05,
             affine=True)
```

主要参数

- num_groups：分组数
- num_channels：通道数（特征数）
- eps：分母修正项
- affine：是否需要affine transform

小结： BN、 LN、 IN和GN都是为了克服Internal Covariate Shift (ICS)

# 十三、Dropout

Dropout：随机失活

随机： dropout probability

失活： weight = 0

<img src="https://s2.loli.net/2022/10/20/UEewtvzNsxV85mc.png" alt="image-20221020182600242" style="zoom: 33%;" />

数据尺度变化：测试时，所有权重乘以 1-drop_prob

---

```py
torch.nn.Dropout(p=0.5, inplace=False)
```

功能： Dropout 层

参数 p：被舍弃概率， 失活概率

实现细节：训练时权重均乘以 $\frac{1}{1-p}$，即除以 $1-p$

一般 Dropout 需要放在需要被 dropout 网络的前面

减轻过拟合，降低方差，控制权重尺度

# 十四、模型保存与加载

## 14.1 序列化与反序列化

```py
torch.save
```

主要参数

- obj：对象
- f：输出路径

```oy
torch.load
```

主要参数

- f：文件路径
- map_location：指定存放位置, cpu or gpu

---

法1: 保存整个Module

```py
torch.save(net, path)
```

法2: 保存模型参数（推荐）

```py
state_dict = net.state_dict() # 保存所有训练好的参数
torch.save(state_dict, path)
```

## 14.2 断点续训练

有时候突然停电，模型突然终止，保存模型参数的机制

```py
checkpoint = {"model_state_dict": net.state_dict(),
              "optimizer_state_dict": optimizer.state_dict(),
              "epoch": epoch
             }
```

# 十五、模型微调

Transfer Learning：机器学习分支，研究源域(source domain)的知识如何应用到目标域(target domain)

模型微调步骤：
1. 获取预训练模型参数
2. 加载模型（load_state_dict）
3. 修改输出层

模型微调训练方法
1. 固定预训练的参数(requires_grad =False； lr=0)
2. Features Extractor较小学习率（ params_group）

# 十六、其他函数

## 16.1 topk

```py
torch.topk(input, k, dim=None, largest=True, sorted=True, out=None)
```

功能：找出前k大的数据，及其索引序号

- input：张量
- k：决定选取 k 个值
- dim：索引维度

返回

- Tensor：前 k 大的值
- LongTensor：前 k 大的值所在的位置













