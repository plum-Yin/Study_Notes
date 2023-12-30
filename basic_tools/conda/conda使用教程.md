# python/package/conda的关系

基于自己对于这些概念术语之间的关系打一个小比方：

- 关于python与package

package =“工具”；

下载 package = “买工具”；

写程序 = "用工具做东西"（程序import导入）

- **关于conda**

环境 = "好比一栋楼，在楼里面分配一间屋给各种‘包’放，每间房里面的‘包’互不影响"

激活环境 = “告诉电脑，我现在要用这个屋子里面的‘包’来做东西了所以要进这间屋子”

移除环境 = “现在这个屋子里面我原来要用的东西现在不需要了把它赶出去节省电脑空间”

Conda创建环境相当于创建一个虚拟的空间将这些包都装在这个位置，我不需要了可以直接打包放入垃圾箱，同时也可以针对不同程序的运行环境选择不同的conda虚拟环境进行运行。

`conda` 是一个环境管理器，当然了你可以不用conda管理环境，python也内置了一些环境管理器，比如说 `env` 和 `virtualenv`。他们有一些很麻烦的缺点。比如说你必须系统环境里面有一个 python 的版本，然后`env `会在你选择的路径下克隆一个纯净的 python，它的版本是你原先系统里面存在的版本，你不能使用这个创建一个新版本。`virtualenv` 好像是可以创建新的版本，但是版本有限，可能你无法下载满意的版本。所以我推荐使用 `conda` 作为环境管理工具。

**例如：**

env1装了pytorch1.0，env2装了pytorch1.2，需要使用1.0的时候激活env1，需要使用pytorch版本1.2的时候激活env2，这样就不用每次配环境一个一个包重新安装。

# conda

## 介绍

通常我们有conda的两种安装方式：Anaconda 和 Miniconda

Anaconda 里面内置了很多软件，比如说 jupyter notebook，Spyder 等。

我觉得 Anaconda 的**优点**是他有一个软件的交互页面，我们可以直接通过**鼠标点击操作**Python的环境管理，版本管理等，当然了我们一般都是通过**命令行**操作这些的。Anaconda 缺点就是安装自带软件较多，磁盘空间占用较多，安装包大小（500MB 左右）是 Miniconda（50MB 左右）接近十倍。

如果你是完全的初学者，电脑的磁盘空间足够（预留 3G 磁盘空间），建议使用 Anaconda，毕竟大部分的东西它已经帮你弄好了。

对于 Miniconda，它的前面有一个 Mini，说明它比较小，它只有一个 conda 环境管理器，没有其他东西，比较干净。（推荐）

## 下载

第一种方式：通过官网下载，直接去搜索引擎搜索 Anaconda3 或者 Miniconda3 既可以。注意：下载的时候后面有一个数字 3，是 python3 的意思，我们用的都是 python3。这个方式有个缺点，那就是由于国内防火墙的问题，下载会很慢。

第二种方式（推荐）：清华镜像。

- Miniconda：https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/
- Anaconda：https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/

对于 Windows 用户，安装的时候特别注意下，某一个页面一定要勾选 `Add to my Path`（差不多意思），这样会把`conda`写入系统环境变量，否则后期在终端中可能找不到`conda`，你需要手动添加。

## 查看环境

打开 Anaconda/Miniconda Prompt，然后输入。如果是 Mac/Linux 直接在 terminal 输入就行。

```bash
conda env list
```

## 创建环境

默认有一个 `base` 主环境，但是不建议使用，因为可能会由于不当操作，导致环境崩掉，软件打不开。所以我们的操作都是在自定义的虚拟环境中操作。

创建一个名为 `demo_env` 的环境，并且使用 `pyhton3.8`

```bash
conda create -n demo_env python=3.8
```

- `-n` 是 name 的简写，后面跟所创建虚拟环境的名字
- 后面是指定python的版本

## 激活环境

```bash
conda activate demo_env
```

## 退出当前环境

退出当前环境默认是返回上一个环境，你可以一直返回，直到退出 conda 环境

```bash
conda deactivate
```

## 删除环境

```bash
conda remove -n demo_env --all
```

- 和创建环境一样，`-n` 后面跟所需要删除的虚拟环境名字
- `--all` 是删除这个环境所有的东西

## 安装 package

有两种方式，一种是使用 `pip`，一种是 `conda`

- `pip` 是 python 自带的一个 package 下载器
- `conda` 是上面安装的

它们之间的安装有一些区别：

- pip是用来安装python包的，安装的是python wheel或者源代码的包。从源码安装的时候需要有编译器的支持，pip也不会去支持python语言之外的依赖项。
- conda是用来安装conda package，虽然大部分conda包是python的，但它支持了不少非python语言写的依赖项，比如mkl cuda这种c c++写的包。然后，conda安装的都是编译好的二进制包，不需要你自己编译。所以，pip有时候系统环境没有某个编译器可能会失败，conda不会。这导致了conda装东西的体积一般比较大，尤其是mkl这种，动不动几百兆甚至一G多。
- 然后，conda功能其实比pip更多。pip几乎就是个安装包的软件，conda是个环境管理的工具。conda自己可以用来创建环境，pip不能，需要依赖virtualenv之类的。意味着你能用conda安装python解释器，pip不行。这一点我觉得是conda很有优势的地方，用conda env可以很轻松地管理很多个版本的python，pip不行。
- 然后是一些可能不太容易察觉的地方。conda和pip对于环境依赖的处理不同，总体来讲，conda比pip更加严格，conda会检查当前环境下所有包之间的依赖关系，pip可能对之前安装的包就不管了。这样做的话，conda基本上安上了就能保证工作，pip有时候可能装上了也不work。不过我个人感觉这个影响不大，毕竟主流包的支持都挺不错的，很少遇到broken的情况。这个区别也导致了安装的时候conda算依赖项的时间比pip多很多，而且重新安装的包也会更多（会选择更新旧包的版本）。
- 最后，pip的包跟conda不完全重叠，有些包只能通过其中一个装。

我一般喜欢用 pip 安装，因为比较快，有些 package 装不好的话再试试 conda，但是不建议在一个虚拟环境中 pip 和 conda 混装 package，可能会出现冲突。

```bash
pip install package_name
conda install package_name
```

也就是 install 后面加 package 的名字就行了。

### package 下载加速（镜像）

由于这些数据库都在境外，国内有防火墙，所以会导致在国内下载非常慢，所以我们需要使用国内的镜像进行加速下载。

#### pip

我们可以使用清华或者豆瓣镜像进行加速下载，也有其他的镜像，这里只列出了这两种。

`-i` 后面跟新镜像的地址

```bash
# （清华镜像）
pip install package_name -i https://pypi.tuna.tsinghua.edu.cn/simple
# 或者（豆瓣镜像）
pip install package_name -i  https://pypi.doubanio.com/simple/ 
```

#### conda

我们需要打开 Anaconda/Miniconda Prompt，然后输入

```bash
# 添加清华镜像
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ 
conda config --set show_channel_urls yes
```

可以用 `conda info` 命令查看当前下载的 channel

如果 `conda install` 仍然出现下载速度慢的错误，

- Windows：可以直接将 `C:/User/用户名` 目录下 `.condarc` 文件 里面的 `-default` 一行删去
- Linux/Mac：`.condarc` 文件默认在 root 目录下（`~/`）

然后正常使用下面命令安装 package 即可。

```bash
conda install package_name
```

### 查看当前环境内的 packages

打开 Anaconda/Miniconda Prompt

```bash
conda list
```



