# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 16:05:31 2021

@author: xiaohuai.le
"""

import pyaudio
import tkinter as tk
import wave
import threading
import queue
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.lines as line
import numpy as np
import soundfile as sf

import socket

#%%
DATA_PORT = 9090
DATA_SIZE = 256
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 连接板子的测试代码
s.connect(('192.168.1.10', DATA_PORT))
# 本机的测试代码
# s.connect(('127.0.0.1', DATA_PORT))

CHUNK = 256 # 从缓存读取音频的长度
N_FFT = 512 # FFT点数
hop = 256   # 滑动窗点数
FORMAT = pyaudio.paInt16 # 读取buffer的形式
CHANNELS = 1             # 读取的通道数量
RATE = 16000             # 读取的采样率
WAVE_OUTPUT_FILENAME = "output.wav"
data =[]
Recording=False
FFT_LEN = 512
frames=[]
counter=1
N = 200         #作图的N的大小
window = np.sqrt(np.hanning(N_FFT))
padding = np.zeros(CHUNK)
MAX = 32767
frame = np.zeros(512)

noisy_s = []
enh_s = []

#GUI
class Application(tk.Frame):
    def __init__(self,master=None):
        tk.Frame.__init__(self,master)
        self.grid()
        self.creatWidgets()

    def creatWidgets(self):
        self.quitButton=tk.Button(self,text='quit',command=root.destroy)
        self.quitButton.grid(column=1,row=3)

# 初始化画布
#make noisy axes and enhance axes
fig = plt.figure()
noisy_ax = plt.subplot(325,xlim=(0,CHUNK*N), ylim=(-MAX,MAX))
enhance_ax = plt.subplot(326,xlim=(0,CHUNK*N), ylim=(-MAX,MAX))
noisy_ax.set_title("noisy signal")
enhance_ax.set_title("enhanced signal")
noisy_line = line.Line2D([],[])
enhance_line = line.Line2D([],[])
#plot data update after reading buffer
noisy_data = np.zeros(CHUNK*N,dtype=np.int16)
enhance_data = np.zeros(CHUNK*N,dtype=np.int16)
noisy_x_data = np.arange(0,CHUNK*N,1)
enhance_x_data = np.arange(0,CHUNK*N,1)

n_stft_ax = plt.subplot(311)
n_stft_ax.set_title("noisy spectrogram")
n_stft_ax.set_ylim(0,N_FFT//2 + 1)
n_stft_ax.set_xlim(0,1000)
n_image_stft = n_stft_ax.imshow(np.random.randn(N_FFT//2 + 1,1000),cmap ='jet')
n_stft_data=np.zeros([257,1000],dtype=np.float32)

stft_ax = plt.subplot(312)
stft_ax.set_title("enhanced spectrogram")
stft_ax.set_ylim(0,N_FFT//2 + 1)
stft_ax.set_xlim(0,1000)
image_stft = stft_ax.imshow(np.random.randn(N_FFT//2 + 1,1000),cmap ='jet')
stft_data=np.zeros([N_FFT//2 + 1,1000],dtype=np.float32)


def overlapadd(frame,hop = 800):
    
    N_frame, L_frame = frame.shape
    length = L_frame + (N_frame - 1) * hop
    output = np.zeros(length)
    for i in range(N_frame):
        output[hop * i : hop * i + L_frame] += frame[i]
    return output 

def plot_init():
    noisy_ax.add_line(noisy_line)
    enhance_ax.add_line(enhance_line)
    return enhance_line,noisy_line,image_stft,n_image_stft 
    
# 画图的更新函数，所有用于更新的数据均被设置为global
def plot_update(i):
    global noisy_data     # 输入音频的数据
    global enhance_data   # 增强音频的数据
    global enhance_x_data # 横轴
    global stft_data      # 增强音频的频谱
    global n_stft_data    # 输入音频的频谱
    noisy_line.set_xdata(noisy_x_data)
    noisy_line.set_ydata(noisy_data)
    
    enhance_line.set_xdata(enhance_x_data)
    enhance_line.set_ydata(enhance_data)
    
    image_stft.set_data(stft_data)
    n_image_stft.set_data(n_stft_data)
    return enhance_line,noisy_line,image_stft,n_image_stft 

# 录音的回调函数
def audio_callback(in_data, frame_count, time_info, status):
    global ad_rdy_ev
    q.put(in_data)
    ad_rdy_ev.set()
    if counter <= 0:
        return (None,pyaudio.paComplete)
    else:
        return (None,pyaudio.paContinue)

#processing block
datad = []
def read_audio_thead(q,stream,frames,ad_rdy_ev):
    global frame 
    while stream.is_active():
        ad_rdy_ev.wait(timeout=1000)
        if not q.empty():
            #process audio data here
            data=q.get()
            while not q.empty():
                q.get()
            # 从buffer获得数据
            noisy_data_0 = np.frombuffer(data,np.dtype('<i2'))
            noisy_s.append(noisy_data_0)
            # 队列移动
            noisy_data[:CHUNK*(N-1)] = noisy_data[CHUNK:CHUNK*N]
            noisy_data[CHUNK*(N-1):] = noisy_data_0
            
            noisy_frame = noisy_data[-FFT_LEN:] / 32768.0 
            noisy_frame = noisy_frame * window
            noisy_fft = np.fft.rfft(noisy_frame)
            # 队列移动
            noisy_spec = np.log((abs(noisy_fft)**2).astype(np.float32)+ 1e-9) + 7 

            n_stft_data[:,:-1] = n_stft_data[:,1:]
            n_stft_data[:,-1] = noisy_spec - 7 
            
            # 发送过去的数据可以是频谱或者是原始音频
            #d = np.round(noisy_spec[1:] / scale).astype(np.int8)
            d = noisy_data_0
            datad.append(d)
            # 其实用int16发送过去就行了
            s.send(d.astype('int32').tobytes())
            # 下面乘以4就是int32的位宽
            recv_data = s.recv(DATA_SIZE * 4)
            # recv_data 就是处理之后的数据，可以是mask也可以是音频
            recv_data = np.frombuffer(recv_data, 'int32')
            """
            mask = recv_data.astype(np.float32)
            
            noisy_fft[1:] = noisy_fft[1:] * mask
            noisy_fft[0] = 0
            enhance_frame = np.fft.irfft(noisy_fft) * window
            frame[CHUNK:] = frame[CHUNK:] + enhance_frame[:CHUNK]
            
            enh_s.append((frame[CHUNK:] * 32768).astype(np.int16))
            """
            stft_data[:,:-1] = stft_data[:,1:]
            stft_data[:,-1] = np.log(abs(np.fft.rfft(frame * window))**2 + 1e-9)      
            # 这里的frame是一个中间变量，你可以认为两个buffer一个frame
            frame[:CHUNK] = frame[CHUNK:]
            frame[CHUNK:] = recv_data
            # 增强音频的队列
            enhance_data[:-CHUNK] = enhance_data[CHUNK:]
            enhance_data[-CHUNK:] = frame[:CHUNK].astype(np.int16)
            enh_s.append(enhance_data[-CHUNK:])
            
        ad_rdy_ev.clear()
        

ani = animation.FuncAnimation(fig, plot_update,
                              init_func=plot_init, 
                              frames=1,
                              interval=30,
                              blit=True)


# pyaudio
p = pyaudio.PyAudio()
q = queue.Queue()
stream = p.open(format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        output=False,
        frames_per_buffer=CHUNK,
        stream_callback=audio_callback)


if Recording:
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)

print("Start Recording")
stream.start_stream()
ad_rdy_ev=threading.Event()

t=threading.Thread(target=read_audio_thead,args=(q,stream,frames,ad_rdy_ev))

t.daemon=True
t.start()

plt.show()
root=tk.Tk()
app=Application(master=root)
app.master.title("Test")
app.mainloop()

stream.stop_stream()
stream.close()
p.terminate()

# 最后你可以把音频保存起来
n = np.concatenate(noisy_s,0)
s = np.concatenate(enh_s,0)
sf.write('./noise_s.wav',n,16000)
sf.write('./enh_s.wav',s,16000)

print("* done recording")

