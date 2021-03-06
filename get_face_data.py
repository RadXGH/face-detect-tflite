import cv2
import os
import time
from os import listdir
import tkinter as tk

def get_face_snapshot():
    global person_name
    faceCascade = cv2.CascadeClassifier('./venv/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')

    # get video capture device
    cam = cv2.VideoCapture(0)
    cam.set(3, 800)
    cam.set(4, 450)

    font = cv2.FONT_HERSHEY_COMPLEX

    # gui
    while True:
        # get video input
        ret, img = cam.read()
        # convert video input color into gray for easier face detection
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # face detection
        box_coords = faceCascade.detectMultiScale(gray_img, 1.1, 4);
        # box on detected face
        for box in box_coords:
            x, y, width, height = box
            x2, y2 = x + width, y + height
            # draw box
            cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0), 2)
            # get the face inside the rectangle
            box_coords = gray_img[y:y2, x:x2]

        # add text
        cv2.putText(img, 'Press q on this window to cancel', (10, 30), font, 0.5, (255, 255, 255), 1)
        cv2.putText(img, 'Press p on this window to take a photo', (10, 50), font, 0.5, (255, 255, 255), 1)
        # add video
        cv2.imshow('Cam', img)

        # press p to take a photo
        if cv2.waitKey(5) == (112 or 80):
            # make new folder for new data
            directory = os.path.join('test-models', person_name)
            os.mkdir(directory)
            # change working directory into the new folder
            os.chdir(directory)
            
            # save the current video frame into the current working directory (take a photo)
            box_coords = cv2.resize(box_coords, (224, 224))
            cv2.imwrite('image.jpg', box_coords)

            # go back up twice in the directory (back to the root dir)
            os.chdir('..')
            os.chdir('..')
            break;
        # press q to quit program
        if cv2.waitKey(1) == (113 or 81):
            break;
    # close video
    cam.release()
    cv2.destroyAllWindows()

def addNewName():
    global person_name
    sameFlag = False
    # get inputted name
    person_name = entry_str.get()
    person_name = person_name.lower()
    entry_str.set('')

    # checks if name is present
    if (person_name != ''):
        # checks if name is already present in dataset
        saved_names = listdir('test-models/')
        for names in saved_names:
            if (person_name == names):
                sameFlag = True
                break
        # open and use the camera to get a video feed for taking a photo
        if (sameFlag == False):
            get_face_snapshot()
    if (sameFlag == False and person_name != ''):
        master.destroy()

# variable declarations
person_name = ''
# create GUI with tkinter
master = tk.Tk()
entry_str = tk.StringVar()

master.title('New name')
tk.Label(master, text = 'Nama lengkap: ').grid(row=0, column=0)
person_name_Panel = tk.Entry(master, textvariable=entry_str, width=30).grid(row=0, column=1, padx=5, pady=5)

tk.Button(master, text = 'Save', width=10, command=addNewName).grid(row=1, column=1, padx=5, pady=5)
tk.Button(master, text = 'Cancel', width=10, command=master.quit).grid(row=1, column=0, padx=5, pady=5)

tk.mainloop()