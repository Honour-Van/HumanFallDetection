# import sysv_ipc as ipc
from cmath import nan
from ctypes import *
# import time
from math import sqrt, atan2
import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from torch.multiprocessing import Lock
import threading
import queue
from scipy.signal import savgol_filter
import time

image_csi = None

class Receiver:
    def __init__(self, draw='amp', subcarrier=20):
        # total received data
        self.tot_recv = []
        self.t = np.zeros(0)
        self.y = np.zeros(0)
        self.draw_mode = 'amp'
        self.subcarrier = subcarrier
        self.port_path='/dev/ttyUSB0'
        self.baudrate=115200
        self.timeout=0.5
        self.data_path='./data.txt'
        self.lock=Lock()
        self.granularity = 10
        self.queue_len = 30


    def get_csi(self, recv_buf: str):
        """
        get_csi:
        handle received data in real time analysis
        @Return:
        a float timestamp, a list of amplitude, a list of phase of the current received data
        @Restriction:
        The data should be in the form of "timestamp,[CSI_DATA]"
        """

        # get raw csi data
        st = recv_buf.index('[')
        ed = recv_buf.index(']')
        raw_csi = recv_buf[st+1:ed-1]
        raw_csi=list(map(int,raw_csi.split())) # str list to int list

        real = raw_csi[::2]  # real part
        img = raw_csi[1::2]  # imaginary part
        length = len(real)
        amp = [sqrt(real[k]**2+img[k]**2) for k in range(length)]
        phs = [atan2(img[k], real[k]) for k in range(length)]

        # get time stamp
        timestamp = float(recv_buf.split(',')[0])
        assert type(timestamp)==float, "timestamp must be float"
        return timestamp, amp, phs

    def segment(self):
        check_queue = self.t[self.queue_len//2*self.granularity:]
        if check_queue.shape[0] < self.granularity:
            return None
        # check_queue = savgol_filter(check_queue, self.granularity, 2)
        var = np.std(check_queue) ** 2
        return var

    def serial_read(self):
        data_file=open(self.data_path,'w+')
        buf = queue.SimpleQueue()
        with serial.Serial(self.port_path,baudrate=self.baudrate,timeout=self.timeout) as ser:
            while True:
                try:
                    last_timestamp = None
                    skip_count = 0

                    while ser.readable():
                        ch = ser.read()
                        if ch == b'\r':
                            continue

                        buf.put(ch)
                        if ch == b'\n':
                            cur_line = []
                            while not buf.empty():
                                cur_line.append(buf.get())
                            cur_line = b''.join(cur_line)
                            cur_data=cur_line.decode(errors='ignore').strip(b'\x00'.decode()).strip('\r')
                            data_file.writelines(cur_data)
                            if ('[' in cur_data) and (']' in cur_data):
                                timestamp, amp, phs = self.get_csi(cur_data)
                                if last_timestamp and (timestamp - last_timestamp > 2 or timestamp < last_timestamp) and skip_count < 3:
                                    skip_count += 1
                                    print("skip once:", timestamp)
                                #     continue
                                    timestamp = float(str(timestamp)[1:])
                                else:
                                    skip_count = 0
                                    last_timestamp = timestamp
                                

                                self.lock.acquire()
                                self.t = np.append(self.t, timestamp)
                                self.y = np.append(self.y, amp[self.subcarrier] if self.draw_mode == 'amp' else phs[self.subcarrier])
                                
                                if self.t.shape[0] // self.granularity > self.queue_len:
                                    self.t = self.t[self.granularity:]
                                    self.y = self.y[self.granularity:]

                                    # self.y[:self.granularity] = savgol_filter(self.y[:self.granularity], self.granularity, 3)
                                seg_vals = self.segment()
                                print(seg_vals)
                                
                                # print(self.t.shape[0])
                                self.lock.release()

                except KeyboardInterrupt:
                    break
                except:
                    continue


    def draw(self, window_size=20):
        plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True

        self.lock.acquire()
        fig = plt.figure()
        fig.canvas.draw()
        plt.plot(self.t, self.y, lw=1)
        self.lock.release()
        if self.t.size==0:
            return
        time_pointer=self.t[-1] 
        if time_pointer > window_size:
            plt.xlim(left=time_pointer-window_size, right=time_pointer)
        # convert canvas to image
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                sep='')
        img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        global image_csi
        image_csi = img
        time.sleep(0.1)
