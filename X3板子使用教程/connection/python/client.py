# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 22:01:57 2022

@author: 乐笑怀
"""


# -*- coding: utf-8 -*-
import socket
 
client = socket.socket()  
client.connect(('127.0.0.1', 9090))        # 设置连接的服务器的IP和端口
 
while True:
    str = input("请输入数据: ")
    client.send(str.encode('utf-8'))       # 设置编码为utf-8
 
client.close()
