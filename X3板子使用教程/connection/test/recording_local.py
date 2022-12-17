import pyaudio
import tkinter as tk
import wave
import socket
import threading
import queue
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import soundfile as sf

# socket_param 
DATA_PORT = 9090
DATA_SIZE = 256
LOCAL_IP = "127.0.0.1"
REMOTE_IP = "192.168.1.10"
ADDRESS = (LOCAL_IP, DATA_PORT)


# pyaudio param
CHUNK = 256              # 从缓存读取音频的长度
FORMAT = pyaudio.paInt16 # 读取buffer的形式
CHANNELS = 1             # 读取的通道数量
RATE = 16000             # 读取的采样率


# stft param
N_FFT = 512 
NUM_BINS = N_FFT // 2 + 1
HOP_LENGTH = 256   # 滑动窗点数
WINDOW = np.sqrt(np.hanning(N_FFT))
PADDING = np.zeros(CHUNK)


# wave param
WAVE_OUTPUT_FILENAME = "./samples/recording_local.wav"

# plt param
N_CHUNK = 200         #作图的N_CHUNK
MAX = 32767
NUM_FRAMES = 1000

counter = 1
mode = "local" # "remote"
recording = False


def overlapadd(frame):
    N_frame, L_frame = frame.shape
    length = L_frame + (N_frame - 1) * HOP_LENGTH
    output = np.zeros(length)
    for i in range(N_frame):
        output[HOP_LENGTH * i : HOP_LENGTH * i + L_frame] += frame[i]
        
    return output 


def plot_init():
    noisy_ax.set_xlim(0, CHUNK*N_CHUNK)
    noisy_ax.set_ylim(-MAX, MAX)
    noisy_ax.set_title("noisy signal")
    
    enhance_ax.set_xlim(0, CHUNK*N_CHUNK)
    enhance_ax.set_ylim(-MAX, MAX)
    enhance_ax.set_title("enhanced signal")
    
    n_stft_ax.set_ylim(0, NUM_BINS)
    n_stft_ax.set_xlim(0, NUM_FRAMES)
    n_stft_ax.set_title("noisy spectrogram")
    
    stft_ax.set_ylim(0, NUM_BINS)
    stft_ax.set_xlim(0, NUM_FRAMES)
    stft_ax.set_title("enhanced spectrogram")
    
    return enhance_line, noisy_line, image_stft, n_image_stft


def plot_update(frame):
    noisy_line.set_xdata(noisy_x_data)
    noisy_line.set_ydata(noisy_data)
    
    enhance_line.set_xdata(enhance_x_data)
    enhance_line.set_ydata(enhance_data)
    
    n_image_stft.set_data(n_stft_data)
    image_stft.set_data(stft_data)
    
    return enhance_line, noisy_line, image_stft, n_image_stft 


# 录音的回调函数
def audio_callback(in_data, frame_count, time_info, status):
    # 存放录音数据
    q.put(in_data) 
    
    if recording:
        wf.writeframes(in_data) 
    
    # event设置为True状态，解除wait
    ad_rdy_ev.set()
    
    if counter <= 0:
        return (None, pyaudio.paComplete)
    
    else:
        return (None, pyaudio.paContinue)


def read_audio():
    while stream.is_active():
        # ad_rdy_ev.wait(timeout=NUM_FRAMES)
        # 个人认为不需设置timeout
        ad_rdy_ev.wait()
        
        if not q.empty():
            # process audio data here
            data = q.get()
            
            while not q.empty():
                q.get()
                
            # 从buffer获得数据 1CHUNK = 256points (<i2: 小端int16)
            noisy_data_0 = np.frombuffer(data, np.dtype('<i2'))
            
            # send 
            # 其实用int16发送过去就行了
            client.send(noisy_data_0.astype("int32").tobytes())
            
            # update noisy_data, 队列左移 1 CHUNK
            noisy_data[:-CHUNK] = noisy_data[CHUNK:]
            noisy_data[CHUNK*(N_CHUNK-1):] = noisy_data_0    
            
            # fft
            noisy_frame = (noisy_data[-N_FFT:] / 32768.0).astype(np.float32)
            noisy_fft = np.fft.rfft(noisy_frame * WINDOW)
            noisy_spec = np.log(abs(noisy_fft)**2+ 1e-9) + 7 

            # update noisy_stft, 队列左移 1 frame
            n_stft_data[:, :-1] = n_stft_data[:,1:]
            n_stft_data[:,-1] = noisy_spec - 7 
            
            # 接收 1 frame 增强后的语音
            # 下面乘以4就是int32的位宽
            recv_data = client.recv(N_FFT * 4)
            recv_data = np.frombuffer(recv_data, 'int32')
            
            # 队列左移 1 CHUNK, overlap-add
            enhance_data[:-CHUNK] = enhance_data[CHUNK:]
            enhance_data[-CHUNK:] = 0
            enhance_data[-N_FFT:] = enhance_data[-N_FFT:] + recv_data.astype(np.int16)
            
            # 更新enhance_stft
            enhance_frame = (recv_data / 32768.0).astype(np.float32)
            enhance_fft = np.fft.rfft(enhance_frame)
            enhance_spec = np.log(abs(enhance_fft) ** 2 + 1e-9) + 7
            
            # 队列左移 1 frame
            stft_data[:, :-1] = stft_data[:, 1:]
            stft_data[:, -1] = enhance_spec - 7
        
                    
        # 清除event的True状态，上方wait使进程阻塞
        ad_rdy_ev.clear()
        
        
def process_audio():
    while stream.is_active():
        # 接收 1CHUNK 数据
        # 下面乘以4就是int32的位宽
        # recv_data 是256点含噪
        recv_data = client_socket.recv(DATA_SIZE * 4)
        recv_data = np.frombuffer(recv_data, 'int32')
        
        # process_data是一个中间变量，代表需要处理的1 frame (512点 int16)
        # update process_data
        process_data[:-CHUNK] = process_data[CHUNK:]
        process_data[-CHUNK:] = recv_data.astype(np.int16)
        
        # 本地测试，process过程简化为加窗
        process_frame = (process_data / 32768.0).astype(np.float32) * WINDOW    
        
        # send process_frame
        send_data = (process_frame * 32768.0).astype("int32")

        client_socket.send(send_data.tobytes())



if __name__ == "__main__":
    # 初始化画布
    # make noisy axes and enhance axes
    fig = plt.figure()
    
    noisy_ax = plt.subplot(325)
    noisy_line, = noisy_ax.plot([], [])
    
    enhance_ax = plt.subplot(326)
    enhance_line, = enhance_ax.plot([], []) 

    n_stft_ax = plt.subplot(311)
    n_image_stft = n_stft_ax.imshow(np.random.randn(NUM_BINS, NUM_FRAMES), cmap='jet')

    stft_ax = plt.subplot(312)
    image_stft = stft_ax.imshow(np.random.randn(NUM_BINS, NUM_FRAMES), cmap='jet')
    
    noisy_x_data = np.arange(0, CHUNK*N_CHUNK, 1)
    noisy_data = np.zeros(CHUNK*N_CHUNK, dtype=np.int16)
    
    enhance_x_data = np.arange(0, CHUNK*N_CHUNK, 1)
    enhance_data = np.zeros(CHUNK*N_CHUNK, dtype=np.int16)

    n_stft_data = np.zeros([NUM_BINS, NUM_FRAMES], dtype=np.float32)
    stft_data = np.zeros([NUM_BINS, NUM_FRAMES], dtype=np.float32)
    
    process_data = np.zeros(N_FFT, dtype=np.int16)


    ani = FuncAnimation(fig=fig, 
                        func=plot_update, init_func=plot_init,
                        frames=1, interval=30, blit=True)

    
    # start client socket
    # client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client = socket.socket()
    
    if mode == "local": 
        # 本机的测试代码
        
        # start server socket
        # server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server = socket.socket()
        server.bind(ADDRESS)
        server.listen(5)
        
        # connect client to server
        client.connect(ADDRESS)
        client_socket, client_addr = server.accept()
    
    # elif mode == "remote":
    #     # 连接板子的测试代码
    #     client.connect((REMOTE_IP, DATA_PORT)) 
        
    else:
        raise NotImplementedError
    
    p = pyaudio.PyAudio()
    stream = p.open(rate=RATE, format=FORMAT, channels=CHANNELS,
                    input=True, output=False,
                    frames_per_buffer=CHUNK, 
                    stream_callback=audio_callback)

    q = queue.Queue()
    
    if recording:
        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
    
    stream.start_stream()
    
    # 读取录音数据
    # 为录音数据回调函数单独创建一个线程，并发执行
    ad_rdy_ev = threading.Event()
    thread_read_audio = threading.Thread(target=read_audio)
    thread_read_audio.setDaemon(True)
    thread_read_audio.start()
    
    # 处理数据
    # 为处理数据过程单独创建一个线程，并发执行
    thread_process_audio = threading.Thread(target=process_audio)
    thread_process_audio.setDaemon(True)
    thread_process_audio.start()
    
    plt.show()
    
    stream.stop_stream()
    stream.close()
    
    if recording:
        wf.close()
    
    p.terminate()

    client_socket.close()
    client.close()
    server.close()

