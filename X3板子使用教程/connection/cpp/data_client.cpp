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
#include "utils.h"

#define DATA_PORT 9090

int main()
{
    const int DATA_SIZE = 512 / 2;
    int sock_fd;
    struct sockaddr_in serv;
    socklen_t servlen;
    int buff[DATA_SIZE];
    sock_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock_fd <= 0)
    {
        perror("creat socket error!");
        return 0;
    }
    perror("Creat socket");
    bzero(&serv, sizeof(serv));
    serv.sin_family = AF_INET;
    serv.sin_port = htons(DATA_PORT);
    serv.sin_addr.s_addr = htons(INADDR_ANY);
    //serv.sin_addr.s_addr = inet_addr("192.168.1.21");
    bzero(&(serv.sin_zero), 8);
    servlen = sizeof(serv);
    while (true)
    {
        if (-1 == sendto(sock_fd, buff, sizeof(buff), 0, (struct sockaddr *)&serv, servlen))
        {
            perror("send data!");
            close(sock_fd);
            return 0;
        }
        if (recvfrom(sock_fd, buff, sizeof(buff), 0, (struct sockaddr *)&serv, &servlen) <= 0)
        {
            perror("recv data!\n");
            close(sock_fd);
            return 0;
        }
    }

    close(sock_fd);
    return 0;
}