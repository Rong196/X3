# -*- coding: utf-8 -*- 
import socket

server = socket.socket()  
server.bind(('127.0.0.1', 9090))                  # 将socket绑定到本机IP并且设定一个端口
server.listen(5)                                  # 设置可以监听5个连接

exit = ''
while True:
    con, addr = server.accept()                   # 会一直等待，直到连接客户端成功
    print('连接到: ', addr)
    while con:
        msg = con.recv(1024).decode('utf-8')      # 接受数据并按照utf-8解码
        print('收到的数据是: ', msg)
        print('收到的数据类型是: ',type(msg))
        if msg == 'break':
                con.close()                       # 关闭本次连接
                exit = 'break'
                break
    
    if exit == 'break':
        break
server.close()                                    # 关闭服务器
