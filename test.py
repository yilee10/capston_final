import matplotlib.pyplot as plt
from collections import deque
import time
import pymysql
import serial
from datetime import datetime
import tensorflow as tf
import numpy as np
from datetime import datetime


#모델 불러오기
loaded_model = tf.keras.models.load_model('final_dense.h5')

# 시리얼 포트와 속도 설정
serial_port = 'COM4'  # 실제 사용하는 포트에 맞게 변경해야 합니다.
baud_rate = 912600

# 시리얼 포트 열기
ser = serial.Serial(serial_port, baud_rate, timeout=1)

#그래프 그리기
graph_time = deque()
Quat_z = deque()
Quat_y = deque()
Quat_x = deque()
Quat_w = deque()
Acc_x = deque()
Acc_y = deque()
Acc_z = deque()
Label = deque()
datas = []


plt.ion()
fig,(ax0,ax1) = plt.subplots(nrows=2, ncols=1,figsize=(15, 8))
line01 , = ax0.plot(graph_time,Quat_w,label='Quatemion_w')
line02 , = ax0.plot(graph_time,Acc_x,label='Acceleration_x')
line03 , = ax0.plot(graph_time,Acc_y,label='Acceleration_y')
line04 , = ax0.plot(graph_time,Acc_z,label='Acceleration_z')
line11 , = ax1.plot(graph_time,Label,label='Label')
ax0.set_ylim([-1.1, 1.1])
ax1.set_ylim([-0.1, 1.1])

i = 0
try:
    while True:
        # 데이터 수신
        received_data = ser.readline().decode('utf-8').strip()
        if received_data:
            now_label = 0
            data_values = received_data.split(',')
            Quatemion_z = float(data_values[1])
            Quatemion_y = float(data_values[2])
            Quatemion_x = float(data_values[3])
            Quatemion_w = float(data_values[4])
            Acceleration_x = float(data_values[-5])
            Acceleration_y = float(data_values[-4])
            Acceleration_z = float(data_values[-3])
            battery = data_values[-2]
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            datas.append([Quatemion_z,Quatemion_y,Quatemion_x,Quatemion_w,Acceleration_x,Acceleration_y,Acceleration_z])
            if(i>100):
                graph_time.popleft()
                Quat_w.popleft()
                Label.popleft()
                Acc_x.popleft()
                Acc_y.popleft()
                Acc_z.popleft()
            if(i>10):
                pred_data = np.array([datas[-10:]])
                pred_result = loaded_model.predict(pred_data,verbose=0)
        
                if (pred_result[0] > 0.5):
                    print("error")
                    now_label = 1
                #else:
                    #print("normal")
                
            graph_time.append(i)
            Quat_w.append(Quatemion_w)
            line01.set_xdata(graph_time)
            line01.set_ydata(Quat_w)

            Acc_x.append(Acceleration_x)
            line02.set_xdata(graph_time)
            line02.set_ydata(Acc_x)

            Acc_y.append(Acceleration_y)
            line03.set_xdata(graph_time)
            line03.set_ydata(Acc_y)

            Acc_z.append(Acceleration_z)
            line04.set_xdata(graph_time)
            line04.set_ydata(Acc_z)

            Label.append(now_label)
            line11.set_xdata(graph_time)
            line11.set_ydata(Label)

            i += 1
        ### 그래프 나타내기 ###
        ax0.relim()
        ax0.autoscale_view()
        ax0.legend()

        ax1.relim()
        ax1.autoscale_view()
        ax1.legend()

        plt.draw()
        plt.pause(0.01)
           
except KeyboardInterrupt:
    # 프로그램 종료 시 시리얼 포트 닫기
    ser.close()
    print("Serial port closed.")
