import socket
import numpy as np

DATA_PORT = 9090
DATA_SIZE = 256
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# 这里举个例子，当需要向板子发送数据时，ip改为192.168.1.10
s.connect(('127.0.0.1', DATA_PORT))

# 我们向端口发送256个随机的int32整数
for i in range(1):
    data = np.random.randint(-50, 51, DATA_SIZE, dtype='int32')
    print('send:',data)
    # 以bytes形式发送就是 b'\x01\x02' 这样子
    s.send(data.tobytes())
    # 以bytes形式接收
    recv_data = s.recv(DATA_SIZE * 4)
    # 转换为np数组
    recv_data = np.frombuffer(recv_data, 'int32')
    print('recieved:', recv_data)
    
s.close()
