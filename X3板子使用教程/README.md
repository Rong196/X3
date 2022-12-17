# X3板子使用教程
@lexiaohuai
## 如何连接板子
- 准备：<br>
1台本电脑，可以使用ssh，有以太网口<br>
一根以太网线<br>
X3板子，X3-pi后说

- 系统安装：(TODO)

  系统需要专门的镜像和软件烧录。已经安装好了，之后我会更新这部分教程

- 如何连接：

  首先X3是运行一个linux的系统的机器，假设系统烧录好了，我们要访问它就和访问服务器一样使用ssh。
  1. 将网线连接开发板网口和电脑网口。
  2. 配置电脑的以太网：打开网络和Internet设置-> 更改适配器选项->以太网->右键属性->选中 Internet协议版本4(TCP/ipv4)→ 属性使用下面的IP地址。若要恢复原局域网络-->选着自动获得IP地址。
  ```
    配置PC的IP为192.168.1.20
    子网掩码设置为255.255.255.0
    网关为192.168.1.255
  ```
  开发板的IP应该是固定`192.168.1.10`。可以通过ssh访问开发板了：
  ```
  ssh root@192.168.1.10
  ```
    flashFXP也可以使用。
     
## 如何把本地的代码编译成在板子上可执行文件

首先开发板上的芯片架构和我们电脑不同，指令集也不同，通常我们必须在开发板上编译出来的文件才能在开发板上运行。如果要在自己电脑上编译然后放在板子上执行，需要使用`交叉编译器`，可以认为是另一个版本的gcc和g++。

目前可以使用的交叉编译器为：`gcc-linaro`，参考官网https://releases.linaro.org/components/toolchain/binaries/

已经有可以使用的安装包：`./gcc-linaro-6.5.0-2018.12-x86_64_aarch64-linux-gnu.tar.xz`，使用时将其解压，放在/opt/下面，这个文件夹里有：`/opt/gcc-linaro-6.5.0-2018.12-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++`和`/opt/gcc-linaro-6.5.0-2018.12-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-gcc`就是c++和c的交叉编译器，可以与使用g++和gcc一样使用它们。

如果要在CMakeLists文件中使用交叉编译，请添加：`set(CMAKE_CXX_COMPILER "/opt/gcc-linaro-6.5.0-2018.12-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++")`

## 如何和板子通信
我们使用socket库进行网络通信，分为clinet用户端和server服务端。我们可以通过python或者C++实现。
使用python向一个端口发送数据，然后板子从这个端口接收。

- 使用python收发的例子：
```
python3 ./connection/python/server.py #服务器启动
python3 ./connection/python/client.py #客户端启动
```
- 使用python发数据，C++接收并返回数据
```
#先编译C++服务器端：
g++ ./connection/test/server.cpp -o server
#运行服务器端
./server
#运行python客户端
python3 ./connection/test/client.py 
```
上述的cpp代码可以运行在板子上，本地可以使用python录音或者发送数据。
当然WSL也可以运行C++脚本接收数据，这样就可以在本地调试了。