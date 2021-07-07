
from tkinter import *
from tkinter import filedialog
import cv2
import os
import numpy as np
from tkinter import messagebox

window=Tk()


data_address = ""
test_address = ""
subjects = [""]



def browsefunc_data():
	global data_address
	data_address = filedialog.askdirectory()


def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
    face_cascade = cv2.CascadeClassifier('haarcascades/lbpcascade_frontalface.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);    
    if (len(faces) == 0):
        return None, None
    (x, y, w, h) = faces[0]    
    return gray[y:y+w, x:x+h], faces[0]

def prepare_training_data(data_folder_path):
    number = 1
    dirs = os.listdir(data_folder_path)
    faces = []
    labels = []    
    for dir_name in dirs:
        subjects.append(dir_name)
        label = number
        number += 1      
        subject_dir_path = data_folder_path + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path) 
               
        for image_name in subject_images_names:            
            if image_name.startswith("."):
                continue            
            image_path = subject_dir_path + "/" + image_name
            image = cv2.imread(image_path)            
            cv2.imshow("Training on image...", image)
            cv2.waitKey(100)            
            face, rect = detect_face(image)            
            if face is not None:
                faces.append(face)
                labels.append(label)            
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    
    return faces , labels


def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)


def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


def predict():
    test_address = filedialog.askopenfilename()
    img = cv2.imread(test_address).copy()
    face, rect = detect_face(img)

    try:
        label, confidence = face_recognizer.predict(face)
    except:
        messagebox.showerror("Error", "Image not compatible try different image")
    label_text = subjects[label]
    
    draw_rectangle(img, rect)
    draw_text(img, label_text, rect[0], rect[1]-5)
    
    cv2.imshow('Output',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()



def answer_return(data_address):
    global face_recognizer
    if data_address == "":
        messagebox.showerror("Error", "Please select proper training dataset folder")
    faces, labels = prepare_training_data(data_address)
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(labels))







label1=Label(window, text="Face recognition", fg='red', font=("Helvetica", 16))
label1.place(x=300, y=20)

label2=Label(window, text="Browse training image folder :", fg='black', font=("Helvetica", 12))
label2.place(x=280, y=100)

label3=Label(window, text="Train data :", fg='black', font=("Helvetica", 12))
label3.place(x=345, y=200)

label4=Label(window, text="Browse test image :", fg='black', font=("Helvetica", 12))
label4.place(x=320, y=300)

browse_btn1=Button(window, text="BROWSE", fg='blue',command=browsefunc_data)
browse_btn1.place(x=355, y=150)

browse_btn2=Button(window, text="TRAIN", fg='white',bg='green',command=lambda: answer_return(data_address))
browse_btn2.place(x=360, y=250)

browse_btn3=Button(window, text="BROWSE & DISPLAY", fg='white',bg='blue',command=predict)
browse_btn3.place(x=325, y=350)

browse_btn4=Button(window, text="EXIT", fg='white',bg='red',command=window.destroy)
browse_btn4.place(x=700, y=350)

window.title('Face recognition')
window.geometry("800x400")
window.mainloop()