[toc]

# 学习目标

- 模型结构设计
  - 小卷积核
  - 堆叠使用卷积核
  - 分辨率减半，通道数翻倍
- 训练技巧
  - 尺度扰动
  - 预训练模型初始化
- 测试技巧
  - 多尺度测试：Dense 测试，Multi-crop 测试
  - 多模型融合

# VGG结构

<img src="https://s2.loli.net/2022/11/24/1Iuegkb2fOA3RFM.png" alt="image-20221125002915138" style="zoom:30%;" />

共性
1. 5个maxpool
2. maxpool后， 特征图通道数翻倍直至512
3. 3个FC层进行分类输出
4. maxpool之间采用多个卷积层堆叠， 对特征进行提取和抽象

Max pool 分辨率减半

---

演变过程

- A：11层卷积 
- A-LRN：基于A增加一个LRN 
- B： 第1，2个block中增加1个卷积 $3*3$ 卷积
- C： 第3， 4， 5个block分别增加1个$1*1$卷积， 表明增加非线性有益于指标提升 
- D：第3， 4， 5个block的$1*1$卷积替换为$3*3$， 
- E：第3， 4， 5个block再分别增加1个$3*3$卷 积

## VGG16 结构

<img src="https://s2.loli.net/2022/11/24/tWJ1pcT4Pv62Hrh.png" alt="image-20221125003129689"  />

 

<img src="https://s2.loli.net/2022/11/24/r5JzNHOl92EDaWS.png" alt="image-20221125003221445" style="zoom: 60%;" />

# VGG 特点

堆叠 $3*3$ 卷积核，增大感受野

- 2 个 $3*3$ 堆叠等价于 1 个 $5*5$
- 3 个 $3*3$ 堆叠等价于 1 个$7*7$

借鉴 NIN，引入利用$1*1$卷积

- 增加非线性激活函数， 增加特征抽象能力，提升模型效果

减少训练参数

可看成$7*7$卷积核的正则化， 强迫$7*7$分解为$3*3$

# 训练技巧

Scale jittering

预训练模型（深层模型用浅层模型初始化，大尺度用小尺度初始化）

## 数据增强

方法一： 针对位置

- 训练阶段：

  1. 按比例缩放图片至最小边为S

  1. 随机位置裁剪出 $224*224$ 区域

  1. 随机进行水平翻转

方法二：针对颜色

- 修改 RGB 通道的像素值， 实现颜色扰动

S设置方法：

1. 固定值： 固定为256， 或384
2. 随机值： 每个 batch 的S在 [256, 512]， 实现尺度扰动

## 预训练模型初始化

深度神经网络对初始化敏感

1. 深度加深时，用浅层网络初始化
  B， C， D， E 用 A 模型初始化
2. Multi-scale训练时， 用小尺度初始化
  - S=384 时， 用 S=256 模型初始化
  - S=[256, 512] 时， 用 S=384 模型初始化

# 测试技巧

多尺度测试

- Dense 测试
- Multi-crop 测试

---

图片等比例缩放至最短边为 Q

设置三个Q， 对图片进行预测， 取平均

- 方法1 当 S 为固定值时：$Q = [S-32, S, S+32]$
- 方法2 当 S 为随机值时：$Q = (S_\min, 0.5*(S_\min + S_\max), S_\max)$

## Dense test

将FC层转换为卷积操作， 变为全卷积网络，实现任意尺寸图片输入

- 利用小卷积核，将输入图片转换成最后输出图像的某一块对应
- 图中最后的 $2\times 2$输出，左上角代表 input 的左上角区域

<img src="https://s2.loli.net/2022/11/06/MUDLnkj1za4tuJi.png" alt="image-20221106170700213" style="zoom: 40%;" />

经过全卷积网络得到 $N*N*1000$ 特征图

在通道维度上求和（sum pool） 计算平均值，得到 $1*1000$ 输出向量

![image-20221127102748038](https://s2.loli.net/2022/11/27/RYbyJSOMtPGmdwZ.png)

## Multi-Crop 测试

借鉴AlexNet与GoogLeNet， 对图片进行Multi-crop， 裁剪大小为$224*224$， 并水平翻转 1 张图， 缩放至 3 种尺寸， 然后每种尺寸裁剪出50张图片；$50 = 5*5*2$

<img src="https://s2.loli.net/2022/11/27/rtdAOGP3e7HzRBD.png" alt="image-20221127104226555" style="zoom:67%;" />

Step1

- 等比例缩放图像至三种尺寸，Q1、Q2、Q3

Step2

- 方法1 Dense：全卷积，sum pool，得到 $1*1000$
- 方法1 Multi-crop：多个位置裁剪 $224*224$ 区域
- 方法1 Multi-crop & Dense：综合取平均

# 实验结果及分析

## Single scale evaluation

S为固定值时：Q = S， S为随机值时：$Q = 0.5(S_\min + S_\max)$

结论：

1. 误差随深度加深而降低， 当模型到达19层时， 误差饱和， 不再下降
2. 增加1*1有助于性能提升
3. 训练时加入尺度扰动， 有助于性能提升
4. B模型中， $3*3$ 替换为 $5*5$ 卷积， top1 下降7%

<img src="https://s2.loli.net/2022/11/27/UMQxeEIsvl6h9mo.png" alt="image-20221127103706245" style="zoom:80%;" />

## Multi-scale evaluation

![image-20221127104347551](https://s2.loli.net/2022/11/27/ze8UWELH6SV3mtZ.png)

方法1：$Q = [S-32, S, S+32]$

方法2：$Q = (S_\min, 0.5*(S_\min + S_\max), S_\max)$

结论：测试时采用 Scale jittering 有助于性能提升

## Multi crop evaluation

方法：等步长的滑动$224*224$的窗口进行裁剪， 在尺度为Q的图像上裁剪$5*5=25$张图片， 然后再进行水平翻转， 得到50张图片， 结合三个Q值，一张图片得到150张图片输入到模型中

结 论 ：

1. mulit-crop 优于 dense
2. multi-crop 结合 dense ， 可形成互补，达到最优结果

![image-20221127104303861](https://s2.loli.net/2022/11/27/wYAaclyC9dJP1Eb.png)

## Convnet fusion

方法:ILSVRC中，多模型融合已经是常规操作

ILSVRC中提交的模型为7个模型融合

采用最优的两个模型

- $D/[256,512]/256,384,512$
- $E/[256,512]/256,384,512$

结合multi-crop和dense，得到最优结果

## Comparison with the state of the art

结论：单模型时，VGG优于冠军GoogLeNet

# 论文总结

关键点&创新点

- 堆叠小卷积核， 加深网络
- 训练阶段， 尺度扰动
- 测试阶段， 多尺度及Dense+Multi crop

启发点
1. 采用小卷积核， 获得高精度
  achieve better accuracy. For instance, the best-performing submissions to the ILSVRC- 2013 (Zeiler & Fergus, 2013; Sermanet et al., 2014) utilized a smaller receptive window size and smaller stride of the
  first convolutional layer. （1 Introduction p2）
2. 采用多尺度及稠密预测， 获得高精度
  Another line of improvements dealt with training and testing the networks densely over the whole image and over multiple scales. (1 Introduction p2)
3. $1*1$ 卷积可认为是线性变换， 同时增加非线性层（Relu）
  In one of the configurations, we also utilize 1 × 1 convolution filters, which can be seen as a linear transformation of the input channels (followed by non-linearity). (2.1 Architecture p1)
4. 填充大小准则： 保持卷积后特征图分辨率不变
  the spatial padding of Conv. layer input is such that the spatial resolution is preserved after convolution (2.1  Architecture p1)
5. LRN对精度无提升
  such normalization does not improve the performance on the ILSVRC dataset, but leads to increased memory consumption and computation time. (2.1 Architecture p3)
6. Xavier初始化可达较好效果
  It is worth noting that after the paper submission, we found that it is possible to initialize the weights without pre-training by using the random initialization procedure of Glorot & Bengio (2010). （3.1 Training p2）
7. S 远大于224， 图片可能仅包含物体的一部分
  $S ≫ 224$ the crop will correspond to a small part of the image, containing a small object or an object part （3.1 Training p4）
8. 大尺度模型采用小尺度模型初始化， 可加快收敛
  To speed up the training of the S = 384 network, it was initialized with the weights pre-trained with S = 256, and we used a smaller initial learning rate of 0.001. (3.1 Training p5)
9. 物体尺寸不一， 因此采用多尺度训练， 可以提高精度
  Since objects in images can be of different sizes, multi-scale training is beneficial to consider during training (3.1 Training p6)
10. multi crop 存在重复计算， 因而低效
  there is no need to sample multiple crops at test time (Krizhevsky et al., 2012), which is less efficient as it requires network re-computation for each crop.(3.2 Testing p2)
11. multi crop可看成dense的补充， 因为它们边界处理有所不同
   Also, multi-crop evaluation is complementary to dense evaluation due to different convolution boundary conditions (3.2 Testing p2)
12. 小而深的卷积网络优于大而浅的卷积网络
   which confirms that a deep net with small filters outperforms a shallow net with larger filters. （4.1 Single scale evaluation p3）
13. 尺度扰动对训练和测试阶段有帮助
   The results, presented in Table 4, indicate that scale jittering at test time leads to better performance（4.2 MMulti-scale evaluation p2）

