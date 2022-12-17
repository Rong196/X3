#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <cstring>

#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <unistd.h>
#include <arpa/inet.h>

#include <ctime>

#define DATA_PORT 9090

int main()
{
  int sock_fd;
  /*local 和 from 是 sockaddr_in结构体：
    struct sockaddr_in {
    short int sin_family;              // Address family    协议，ipv4或者ipv6等
    unsigned short int sin_port;       // Port number       端口号
    struct in_addr sin_addr;           // Internet address  地址
    unsigned char sin_zero[8];         // Same size as struct sockaddr 
    };
    */
  struct sockaddr_in local;
  struct sockaddr_in from;
  socklen_t fromlen;
  // 这里我们设置DATA_SIZE为512,即一次性传入512个数据
  const int DATA_SIZE = 512;
  const int taps = 2;
  // 传入和传出的数据被保存在DATA_SIZE数组中，类型是int32，即一共512*4=2048字节。当然你也可以选择传入float
  int recv_buff[DATA_SIZE];
  int send_buff[DATA_SIZE];
  // 使用std::memset把2048个字节地址中所有的数据初始化为0。
  std::memset(recv_buff, 0, sizeof(recv_buff));
  std::memset(send_buff, 0, sizeof(send_buff));

  // int8_t BUFF[DATA_SIZE];
  // 创建socket
  sock_fd = socket(AF_INET, SOCK_DGRAM, 0);
  if (sock_fd <= 0)
  {
    perror("creat socket error!");
    return 0;
  }
  perror("Creat socket");

  /*设置server地址结构*/
  bzero(&local, sizeof(local));
  local.sin_family = AF_INET;
  local.sin_port = htons(DATA_PORT);
  local.sin_addr.s_addr = htons(INADDR_ANY); // 设置ip， INADDR_ANY是任意ip，如果要固定ip，可以使用下面的代码
  //local.sin_addr.s_addr = inet_addr("127.0.0.1");
  bzero(&(local.sin_zero), 8);

  if (0 != bind(sock_fd, (struct sockaddr *)&local, sizeof(local)))
  {
    perror("bind address fail!\n");
    close(sock_fd);
    return 0;
  }
  printf("bind socket!");

  fromlen = sizeof(from);
  printf("waiting request from client...\n");

  while (true)
  {
    // 下面的代码是用std::memove实现把一半的数据从recv_buff后半部分移动到开头，通常音频是一块一块输入的，首先需要这种的队列。
    // std::memmove(recv_buff, recv_buff + DATA_SIZE - DATA_SIZE / taps, sizeof(recv_buff) - sizeof(recv_buff) / taps);
    // 从端口接收数据放到recv_buff后半部分
    if (recvfrom(sock_fd, recv_buff + DATA_SIZE - DATA_SIZE / taps, sizeof(recv_buff) / taps, 0, (struct sockaddr *)&from, &fromlen) <= 0)
    {
      perror("recv data error!\n");
      close(sock_fd);
      return 0;
    }
    // TODO, 你在这边对数据进行处理
    // Pre_processing(recv_buff);
    // 举个例子，就把接收到的数据输出出来
    std::cout << "recieved:" << std::endl;
    for (int i = 0; i < DATA_SIZE / taps; i++)
    {
        std::cout << *(recv_buff + DATA_SIZE - DATA_SIZE / taps + i) << " ";
    }
    std::cout << std::endl;
    
    // TODO, 你在这边把输出处理
    // Post_processing(output);
    // 举个例子，我就把得到的输入数据复制给输出的send_buff
    std::memmove(send_buff, recv_buff + DATA_SIZE - DATA_SIZE / taps, sizeof(recv_buff) - sizeof(recv_buff) / taps);

    if (sendto(sock_fd, send_buff, sizeof(send_buff) / taps, 0, (struct sockaddr *)&from, fromlen) == -1)
    {
      perror("send data error!\n");
      close(sock_fd);
      return 0;
    }
    std::memmove(send_buff, send_buff + DATA_SIZE / taps, sizeof(send_buff) - sizeof(send_buff) / taps);
    printf("send data!\n");
  }

  close(sock_fd);

  return 0;
}
