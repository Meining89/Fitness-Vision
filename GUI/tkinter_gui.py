
import numpy as np
import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import ImageTk, Image



# 调用Tk()创建主窗口
root_window =tk.Tk()
# 给主窗口起一个名字，也就是窗口的名字
root_window.title('Fitness Vision')
# 设置窗口大小:宽x高,注,此处不能为 "*",必须使用 "x"
root_window.geometry('1000x700')

#Graphics window
imageFrame = tk.Frame(root_window, width=600, height=500)
imageFrame.grid(row=0, column=0, padx=10, pady=2)

#Capture video frames
lmain = tk.Label(imageFrame)
lmain.grid(row=0, column=0)
cap = None  # Will store the capture object

def open_camera():
    global cap
    cap = cv2.VideoCapture(0)  # Open the default camera (camera index 0)
    text.grid_forget()  # Hide the welcome label
    description.grid_forget()  # Hide the description label
    show_frame()  # Start displaying frames

def show_frame():
    if cap is not None:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(10, show_frame) 
    else:
        print("Camera not open")



#Slider window (slider controls stage position)
sliderFrame = tk.Frame(root_window, width=600, height=100)
sliderFrame.grid(row = 1, column=0, padx=10, pady=2)

# add button: open camera/opencv
button=tk.Button(root_window,text="Start",command=open_camera)

# def callback():

# button=tk.Button(root_window,text="Start",command=callback)
# 将按钮放置在主窗口内
button.grid(row=2, column=0, pady=10)  # Use grid instead of pack


# 更改左上角窗口的的icon图标,加载C语言中文网logo标
#root_window.iconbitmap('C:/Users/Administrator/Desktop/favicon.ico')

# # 设置主窗口的背景颜色,颜色值可以是英文单词，或者颜色值的16进制数,除此之外还可以使用Tk内置的颜色常量
# root_window["background"] = "#C9C9C9"

# 定义回调函数，当用户点击窗口x退出时，执行用户自定义的函数
def QueryWindow():
    # 显示一个警告信息，点击确后，销毁窗口
    if messagebox.showwarning("Warning","Data will not be saved"):
        if cap is not None:
            cap.release()  # Release the camera capture
        # 这里必须使用 destory()关闭窗口
        root_window.destroy()


# # 添加文本内,设置字体的前景色和背景色，和字体类型、大小
text=tk.Label(root_window,text="Welcome to Fitness Vision",bg="blue",fg="white",font=('Times', 20, 'bold italic'))
# # 将文本内容放置在主窗口内
text.grid(row=0, column=0, pady=10)
description = tk.Label(root_window,text="Your Real-time squate counter & evaluation assistant",bg="blue",fg="white",font=('Times', 20, 'bold italic'))
description.grid(row=1, column=0, pady=10)




# 使用协议机制与窗口交互，并回调用户自定义的函数
root_window.protocol('WM_DELETE_WINDOW', QueryWindow)

#开启主循环，让窗口处于显示状态
root_window.mainloop()