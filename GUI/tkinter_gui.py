
import numpy as np
import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import ImageTk, Image

def open_camera():
    global cap
    cap = cv2.VideoCapture(0)  # Open the default camera (camera index 0)
    text.grid_remove()  # Hide the welcome label
    description.grid_remove()  # Hide the description label
    button.grid_remove()  # Hide the button
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


# 调用Tk()创建主窗口
root_window =tk.Tk()
# 给主窗口起一个名字，也就是窗口的名字
root_window.title('Fitness Vision')
# Get the screen width and height
screen_width = root_window.winfo_screenwidth()
screen_height = root_window.winfo_screenheight()
# Set the window size to fill up the screen
root_window.geometry(f'{screen_width}x{screen_height}')

# Graphics window
imageFrame = tk.Frame(root_window, width=int(screen_width * 0.75), height=screen_height)  # Adjusted width
imageFrame.grid(row=0, column=0, padx=10, pady=2, sticky='nsew')  # Sticky to fill the available space


#Capture video frames
lmain = tk.Label(imageFrame)
lmain.grid(row=0, column=0)
cap = None  # Will store the capture object


# # 添加文本内,设置字体的前景色和背景色，和字体类型、大小
text=tk.Label(root_window,text="Welcome to Fitness Vision",bg="blue",fg="white",font=('Times', 20, 'bold italic'))
text.grid(row=0, column=0, pady=10, sticky='n')  # Align to the top
description = tk.Label(root_window,text="Your Real-time squate counter & evaluation assistant",bg="blue",fg="white",font=('Times', 20, 'bold italic'))
description.grid(row=1, column=0, pady=10, sticky='n')

# add button: open camera/opencv
button=tk.Button(root_window,text="Start",command=open_camera)

button.grid(row=4, column=0, pady=10, sticky='n')  



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







# 使用协议机制与窗口交互，并回调用户自定义的函数
root_window.protocol('WM_DELETE_WINDOW', QueryWindow)

#开启主循环，让窗口处于显示状态
root_window.mainloop()