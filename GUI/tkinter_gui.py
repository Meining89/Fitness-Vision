import numpy as np
import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import ImageTk, Image
import customtkinter

def open_camera():
    global cap
    cap = cv2.VideoCapture(0)  # Open the default camera (camera index 0)

    text.grid_remove()  # Hide the welcome label
    description.grid_remove()  # Hide the description label
    button.grid_remove()  # Hide the button
    imageFrame.config(height=root_window.winfo_screenheight(), width=0.75*root_window.winfo_screenwidth())  # Adjust height and width dynamically
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 0.8*root_window.winfo_screenwidth())  # Set the camera frame width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 0.8*root_window.winfo_screenheight())  # Set the camera frame height
    
    # Reconfigure the grid to move imageFrame to the left
    root_window.grid_columnconfigure(0, weight=0)  # Remove weight from column 0
    root_window.grid_columnconfigure(1, weight=1)  # Add weight to column 1

    # Reconfigure the grid to move imageFrame to the left
    imageFrame.grid(row=0, column=0, padx=10, pady=2, sticky='w')  # Stick to the left
    
    show_frame()  # Start displaying frames

def show_frame():
    if cap is not None:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (int(0.8*root_window.winfo_screenwidth()), int(0.8*root_window.winfo_screenheight())))  # Resize the frame
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)

        # Update lmain with the new image
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)

        # Schedule the next frame update
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
root_window.grid_rowconfigure(0, weight=1)
root_window.grid_columnconfigure(0, weight=1)

# Graphics window
imageFrame_width = int(screen_width)
imageFrame = tk.Frame(root_window, width=imageFrame_width, height=screen_height)  # Adjusted width
imageFrame.grid(row=0, column=0, padx=10, pady=2)  # Sticky to fill the available space


#Capture video frames
lmain = tk.Label(imageFrame)
lmain.grid(row=0, column=0)
cap = None  # Will store the capture object


# welcome message
text=tk.Label(root_window,text="Welcome to Fitness Vision",fg="black",font=('Roboto', 30, 'bold '))
text.grid(row=0, column=0, pady=10, sticky='n')  # Align to the top
description = tk.Label(root_window,text="Your Real-time squat counter & evaluation assistant",fg="black",font=('Roboto', 20, 'bold '))
description.grid(row=0, column=0, pady=80, sticky='n')

# add button: open camera/opencv
# button=tk.Button(root_window,text="Start",command=open_camera)
# use custom tkinter instead
button=customtkinter.CTkButton(root_window,text="Start",command=open_camera)

button.grid(row=0, column=0, pady=150, sticky='n')  



def QueryWindow():
    if messagebox.showwarning("Warning","Data will not be saved"):
        if cap is not None:
            cap.release()  # Release the camera capture
       
        root_window.destroy()


root_window.protocol('WM_DELETE_WINDOW', QueryWindow)

root_window.mainloop()