# jetson nano b01 开发

## （一）**开发环境配置**
### （1）__系统烧录__
* 下载 VMware ，导入商家配套的虚拟机镜像文件。由于镜像文件是分卷压缩的，所以解压时候只需要解压卷号为 001 的即可得到全部内容，解压软件为 7-zip 。  
打开 VMvware ,打开虚拟机，在解压得到的文件夹中选择虚拟机配置文件。输入密码 **123456** 进行登录。

* 将 jetson nano 的镜像文件烧录到 SD 卡。解压商家配套的 jetson nano 镜像文件，同样解压卷 **001** 即可，然后打开烧录软件（Win32DiskImager.7z），导入得到的 img 或 zip 文件，完成烧录即可。

* 接入显示屏查看。商家设置的初始登录账号和密码都是 **nvidia** ，首次登录后最好设置一下屏幕关闭时间以及关闭登录密码，方便后续使用。  
如果有必要还可以下载中文输入法，中文输入法安装完成后注意要设置 `输入源` ，否则可能无法使用。但是不建议安装，太占内存。

* jetson nano 供电方式主要有 DC（5 V / 4 A ） 和 USB（5 V / 2 A ） 。使用 DC 时要将电源接口附近的跳线帽按下，使用 USB 则拔出。

### （2）**文件传输软件 FileZilla**
* 在笔记本上安装商家配套的 FileZilla_win64 ，将笔记本电脑和开发板使用 USB 数据线连接，**且确保两者连接同一网络**，否则不能传输文件。  
然后在开发板上打开终端，输入 `hostname -I` 查看 ip 地址，在不同的网络中开发板被分配的 ip 地址可能不一样，本机在当前的网络是 `192.168.10.24` ，用户名和密码都是 `nvidia` ，打开 FileZilla ，输入 ip 、用户名和密码，连接之后即可传输文件。

### （3）**jetson nano 自带环境检查**
* jetson nano 内置了 python、CUDA 等。

* python 。有 python2 和 python3 ，打开终端，输入 `pyhton --version` ，显示的是默认版本 python2 。  
输入 `pyhton3 --version` ，显示的是 python3 的版本，输入 python3 即可打开 python3 ，打开后输入下面的代码，即可查看所安装的 opencv 版本。最后输入 `quit()` 退出 python 。
```python
import cv2
cv2.__version__
``` 
* CUDA 。在 usr/local 文件夹中有 CUDA 文件夹，其中可查看版本。
* cudnn 。在 usr/include 文件夹中。

### （4）**远程控制和命令行连接 WiFi**
* **远程控制方法 1** ：先插上电源线，通电启动后再插上 USB 数据线，（电源线和数据线两个一起插上，再通电启动的话不行，电源灯不亮。）打开商家配套辅助软件中的 putty 软件，连接方式选择 Serial ，输入 COM3 ，使用系统的 speed ，点击 open 打开。输入 Linux 账号和密码，本机账号密码都是 nvidia ，连接成功，可以远程操控开发板。  
如果关闭当前的 putty 窗口，重新打开再连接，虽然是显示空屏，但默认是连接上的，按 enter 即会显示当前目名。  
**远程控制方法 2** ：确保 nano 开发板和控制设备连接同一个网络，我是连接的 WiFi 。  
打开商家配套辅助软件中的 putty 软件，输入 ip 地址（我的是 192.168.10.24），选择 SSH 连接，然后打开。输入 Linux 账号和密码，本机账号密码都是 nvidia ，连接成功，可以远程操控开发板。  
**命令行连接 WiFi**：
```
# 1、首先要插上 WiFi 模块，不然无法识别并连接 WiFi
# 2、开机，打开终端，输入以下命令

# 查看网络设备
sudo nmcli dev
# 开启 WiFi
sudo nmcli r wifi on
# 扫描附近 WiFi
sudo nmcli dev wifi
# 连接 wifi
sudo nmcli dev wifi connect 要连接的WiFi名 password 要连接的WiFi密码

# 3、连接 WiFi 整个过程的主要命令如下
# 开启 WiFi
sudo nmcli r wifi on
# 扫描附近 WiFi
sudo nmcli dev wifi
# 连接 wifi
sudo nmcli dev wifi connect 要连接的WiFi名 password 要连接的WiFi密码
# 比如 WiFi 名是 DSC1998 ，密码是 987654321 ，则输入以下命令
sudo nmcli dev wifi connect DSC1998 password 987654321

# 4、测试网络是否接通
# ping 百度看看能否 ping 通
ping www.baidu.com

```

### （5）**vscode 安装**
* 为了更好地开发，建议在开发板上安装 arm 版本的 vscode ，下载网页：<https://code.visualstudio.com/docs/supporting/faq#_previous-release-versions>，由于操作系统是 ubuntu ，所以选择 Linux Arm64 debian 的安装包进行下载。  
具体安装包下载地址：<https://update.code.visualstudio.com/1.85.1/linux-deb-arm64/stable>，其中 `1.85.1` 是版本号，可自行修改，但是由于开发板的 ubuntu 版本较低，不支持最新版的 vscode ，所以尽量安装低一些的版本。  
百度网盘下载：https://pan.baidu.com/s/1p2ZCUFz0uKfNQoPsekhbQg 提取码: cpdd

* 打开终端，输入 `sudo dpkg -i 包名.deb` ，这里下载的包名是 code_1.85.1-1702461056_arm64 ，所以输入 `sudo dpkg -i code_1.85.1-1702461056_arm64.deb` ，等待安装完成即可。对应的卸载命令是 `sudo apt-get remove code` ，其中 code 是软件名，这种方法不删除配置文件。 

* 打开 vscode ，首先安装 python 插件，然后新建一个 python 文件，将其打开后在界面右下角选择解释器 python3.6.9 64-bit ，不要选择 python2 。  
设置主题和字体，这里的字体一般是代码编辑窗口的字体、字号，使用 `Ctrl +` 和 `Ctrl -` 可分别实现整个窗口的字号加、减。  
打开设置，搜索 completeFunctionParens ，打开函数括号自动补全，还可以下载一些代码自动补全的插件。 搜索 format（格式化），打开“保存时格式化文件”
配置完成之后，由于开发板自带了一些如 CUDA、tensorRT 的库，再安装一些如 torch、torchvision 的库即可方便地进行开发。

### （5）**刷机教程**
* 刷机时只需要重新将 SD 卡的内容进行烧录即可，不需要再使用虚拟机镜像。  
注意首先需要将 SD 卡格式化，具体步骤自行搜索，不然插入电脑可能无法识别，然后才能进行烧录操作。

### （6）**调用摄像头**
* 安装 v4l2-utils 协助工具：`sudo apt install v4l-utils `，  
查看摄像头挂载情况：`ls /dev/video*`，  
查看所挂载的 video0 摄像头参数：`v4l2-ctl --device=/dev/video0 --list-formats-ext`，  
检测摄像头能否正常打开：`nvgstcapture`，
* 使用 OpenCV 调用 CSI 摄像头，注意CSI 摄像头不能直接在开发板带电的情况下进行拔插，有烧毁的风险。  
Jetson Nano 默认安装了 OpenCV 4.1.1 版本，可以利用 Gstreamer 或 Jetcam 打开 CSI 摄像头，介绍 Gstreamer 方法，首先需要安装 Gstreamer ，代码如下，在终端中依次执行代码：
```
sudo add-apt-repository universe
sudo add-apt-repository multiverse
sudo apt-get update

sudo apt-get install gstreamer1.0-tools gstreamer1.0-alsa gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav

sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-good1.0-dev libgstreamer-plugins-bad1.0-dev

```

完整的读取代码如下，假设该代码所在的文件名为 CSI_cam.py ，那么在该文件所在的文件夹打开终端后，输入 `python3 CSI_cam.py` 即可执行该文件，实现打开摄像头的效果。
```python
import CV2

# 设置gstreamer管道参数
def gstreamer_pipeline(
    capture_width=1280, #摄像头预捕获的图像宽度
    capture_height=720, #摄像头预捕获的图像高度
    display_width=1280, #窗口显示的图像宽度
    display_height=720, #窗口显示的图像高度
    framerate=60,       #捕获帧率
    flip_method=0,      #是否旋转图像
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

if __name__ == "__main__":
    capture_width = 1280
    capture_height = 720

    display_width = 1280
    display_height = 720

    framerate = 60			# 帧数
    flip_method = 0			# 方向

    # 创建管道
    print(gstreamer_pipeline(capture_width,capture_height,display_width,display_height,framerate,flip_method))

    #管道与视频流绑定
    cap = CV2.VideoCapture(gstreamer_pipeline(flip_method=0), CV2.CAP_GSTREAMER)

    if cap.isOpened():
        window_handle = CV2.namedWindow("CSI Camera", CV2.WINDOW_AUTOSIZE)

        # 逐帧显示
        while CV2.getWindowProperty("CSI Camera", 0) >= 0:
            ret_val, img = cap.read()
            CV2.imshow("CSI Camera", img)

            keyCode = CV2.waitKey(30) & 0xFF
            if keyCode == 27:# ESC键退出
                break

        cap.release()
        CV2.destroyAllWindows()
    else:
        print("打开摄像头失败") 

```

* 调用 USB 摄像头。可直接调用，代码如下。需要连接了 USB 摄像头，并且填入其正确的设备号 dev ：
```python
import cv2

cap = cv2.VideoCapture(dev)     # dev表示Jetson Nano上USB的设备号，上面查看摄像头挂载情况那里提到我的USB是video2，故我这里dev = 2

while(1):
    ret,frame = cap.read()  #ret:True/False,代表有没有读到图片  frame:当前截取一帧的图片
    cv2.imshow("capture",frame) 
   
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break  
    
cap.release()
cv2.destroyAllWindows()
```

## （二）深度学习模块配置
### （1）**jetson-inference 库安装**

* 删除 ubuntu 系统上的示例文档，增加空间，代码如下。还可以卸载 libreoffice 。
```
sudo dpkg -r --force-depends "cuda-documentation-10-2" "cuda-samples-10-2" "libnvinfer-samples" "libvisionworks-samples" "libnvinfer-doc" "vpi1-samples"
```
* 下载 jetson-inference 库。需要从 github 上下载，但是开发板不能访问 github ，需要进行设置。  
而且从 Windows 上下载再传到开发板也不行，会丢东西导致编译错误。  
所以需要修改开发板上操作系统的 hosts 文件，步骤如下：  
1. 打开 DNS 查询网站：<https://tool.chinaz.com/dns/?type=1&host=github.com&ip=>
2. 输入 github 的域名：github.com 进行检测。
3. 选择 TTL 值较小的 IP 地址，复制到 hosts 文件中。
```
# 使用 vim 打开 hosts 文件中
sudo vim /etc/hosts

# 添加 ip 地址和域名，比如
20.205.243.166 github.com
```
4. 刷新 DNS ，然后就可以访问 github 了
```
sudo service network-manager restart
```

* 下载 jetson-inference 库。打开终端，依次输入执行以下代码。
```
# 先安装依赖包
sudo apt-get update
sudo apt autoremove
sudo apt-get install git cmake libpython3-dev python3-numpy

# 克隆仓库到本地，--recursive 是递归下载子模块，--depth=1 是下载最新提交的内容
# 注意下载完后要显示子模块也下载完全才行，显示 fail 则不行。
git clone --recursive --depth=1 https://github.com/dusty-nv/jetson-inference

# 进入克隆的 jetson-inference 目录，
# 这个代码是更新子模块，由于使用了 --depth=1 进行克隆，所以不需要再执行。
# git submodule update --init 

# 进入 tools 文件夹，换源
sed -in-place -e 's@https://nvidia.box.com/shared/static@https://bbs.gpuworld.cn/mirror@g' download-models.sh
sed -in-place -e 's@https://nvidia.box.com/shared/static@https://bbs.gpuworld.cn/mirror@g' install-pytorch.sh

# 返回 jetson-inference 文件夹，创建 build 文件夹中
mkdir build

# 进入 build 文件夹中
cd build

# 运行cmake，它会执行上一级目录下面的 CMakePrebuild.sh
cmake ../

# cmake 时可能会弹出选择模型下载和 pytorch 安装的窗口，
# 模型不选，跳过，pytorch 则选择 python3.6 的进行安装，也可以不安装，反正下载不了。

# 开始编译
make

# 如果编译时出错，一般是子模块没有下载完全，重新下载或是手动下载缺失的包。

# 将程序安装至系统中。如果原始码编译无误，且执行结果正确，便可以把程序安装至系统预设的可执行文件存放路径。默认/usr/local/bin
sudo make install

# 编译安装完成后 build 文件夹中会有 aarch64/bin 文件夹，示例程序和图像在这里。
```

* 由于 nvidia.box.com 不能访问，所以 pytorch 需要下载安装包自行安装，  
本机对应的版本是 Python 3.6 - torch-1.6.0-cp36-cp36m-linux_aarch64.whl ，
百度网盘链接：https://pan.baidu.com/s/1-CZ9rwABuOCPbdXjU0SzbA 提取码: cpdd  
下载后直接使用 pip3 安装，pip3 未安装则先进行安装，
```
# pip3 install 包名，如下：
pip3 install torch-1.6.0a0+b31f58d-cp36-cp36m-linux_aarch64.whl
```
安装完成后在任意终端输入 `python3` 来打开 python3 ，输入 `import torch` 可能会出现  
`Illegal instruction(core dumped)` 错误，需要编辑环境变量解决方法如下：
```
# 使用 vim 打开环境变量文件
sudo vim ~/.bashrc

# 打开后按 'i' 进入插入模式，在最后一行插入 export OPENBLAS_CORETYPE=ARMV8
# 按 'esc' 进入返回命令模式，输入 ':wq' 进行保存和退出

# 最后重新激活环境变量
source ~/.bashrc
```

完成后在任意终端输入 `python3` 来打开 python3 ，输入 `import torch`，  
此时可以正常导入 torch ，再输入 `torch.cuda.is_available()` ，  
输出结果应该是 True ，表示可以使用 GPU ，pytorch 安装完成。

### （2）jetson-inference 需要的预训练模型下载
预训练模型包括分类、目标检测、分割等网络，根据实际需求下载。
到 github 下载需要的模型，链接：<https://github.com/dusty-nv/jetson-inference/releases>  
百度网盘下载 ResNet-18 ：https://pan.baidu.com/s/1iLTWJh62NMcAt6J1HmyqTQ 提取码: cpdd  
将下载的模型压缩包复制到 jetson-inference/data/networks 中，然后使用 `tar -zxvf 包名.tar.gz` 进行解压，  
比如 `tar -zxvf ResNet-18.tar.gz` ，奇怪的是 ResNet-18 解压后的内容在一个文件夹中，在 vscode 中可以加载出来使用，  
但是 GoogleNet 解压后的内容不是在一个文件夹中，而且加载时却显示还要从外网下载，无法使用。经过后续摸索找到解决办法：在 jetson-inference/data/networks 中创建一个名为 Googlenet 的文件夹，注意只有 G 是大写，其它都小写，然后将 GoogleNet.tar.gz 解压到这个文件夹中，即可使用 vscode 加载。

## （三）其它配置
### （1）关闭桌面模式
桌面模式占用内存，熟悉之后可切换到命令行模式。
切换到命令行模式的代码，重启后生效。
```
sudo systemctl set-default multi-user.target
```
切换到桌面模式的代码，重启后生效。
```
sudo systemctl set-default graphical.target
```