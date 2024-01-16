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


#first window
root_window =tk.Tk()
root_window.title('Fitness Vision')
# Get the screen width and height
screen_width = root_window.winfo_screenwidth()
screen_height = root_window.winfo_screenheight()
# Set the window size to fill up the screen
root_window.geometry(f'{screen_width}x{screen_height}')

# Graphics window
imageFrame = tk.Frame(root_window, width=int(screen_width * 0.5), height=screen_height)  # Adjusted width
imageFrame.grid(row=0, column=0, padx=10, pady=2, sticky='nsew')  # Sticky to fill the available space


#Capture video frames
lmain = tk.Label(imageFrame)
lmain.grid(row=0, column=0)
cap = None  # Will store the capture object


# welcome message
text=tk.Label(root_window,text="Welcome to Fitness Vision",bg="blue",fg="white",font=('Times', 20, 'bold italic'))
text.grid(row=0, column=0, pady=10, sticky='n')  # Align to the top
description = tk.Label(root_window,text="Your Real-time squate counter & evaluation assistant",bg="blue",fg="white",font=('Times', 20, 'bold italic'))
description.grid(row=1, column=0, pady=10, sticky='n')

# add button: open camera/opencv
button=tk.Button(root_window,text="Start",command=open_camera)

button.grid(row=4, column=0, pady=10, sticky='n')  



def QueryWindow():
    if messagebox.showwarning("Warning","Data will not be saved"):
        if cap is not None:
            cap.release()  # Release the camera capture
       
        root_window.destroy()


root_window.protocol('WM_DELETE_WINDOW', QueryWindow)

root_window.mainloop()