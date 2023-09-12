from PIL import Image
import os
from tkinter import *
from tkinter.ttk import *
from tkinter import ttk
from tkinter import messagebox
import tkinter
import cv2
from PIL import ImageTk, Image
import numpy as np
from facenet_pytorch import MTCNN
import torch
from predicter.predict import Mtcnn_face_predict, Mtcnn_face_detect
from facenet_pytorch import MTCNN

RUN = True
run_detect = 0

def _from_rgb(rgb):
    """translates an rgb tuple of int to a tkinter friendly color code
    """
    return "#%02x%02x%02x" % rgb


count_nhandien = -100
face_dict = {}
capture = 0

win = Tk()
win.title("Nhận diện khuôn mặt")
win.geometry("1000x700")

# CONFIG
# PATH
base_dir = os.path.dirname(__file__)
# COLOR
lightGray = _from_rgb((240, 240, 240))
white = _from_rgb((255, 255, 255))

# FONT
font_header1 = "Arial 20 bold"
font_header2 = "Arial 16 bold"
font_content = "Arial 12"

# IMAGE
bg_image = Image.open("imageGUI/bg_app.jpg")
bg_image = bg_image.resize(
    (1000, 700), Image.ANTIALIAS)
bg_image = ImageTk.PhotoImage(bg_image)

default_them_nguoi = Image.open(base_dir+"/imageGUI/default_Image.png")
default_them_nguoi = default_them_nguoi.resize(
    (560, int(3*560/4)), Image.ANTIALIAS)
default_them_nguoi = ImageTk.PhotoImage(default_them_nguoi)

default_empty = Image.open(base_dir+"/imageGUI/default_empty.png")
default_empty = default_empty.resize(
    (60, 60), Image.ANTIALIAS)
default_empty = ImageTk.PhotoImage(default_empty)

button_them_nguoi = Image.open("imageGUI/button_them_nguoi.png")
button_them_nguoi = button_them_nguoi.resize(
    (60, 60), Image.ANTIALIAS)
button_them_nguoi = ImageTk.PhotoImage(button_them_nguoi)

arow = Image.open("imageGUI/arow.png")
arow = arow.resize(
    (160, 80), Image.ANTIALIAS)
arow = ImageTk.PhotoImage(arow)
# End config

trang_chu = tkinter.Frame(win)
nhan_dien = tkinter.Frame(win)
them_nguoi = tkinter.Frame(win)

frames = (trang_chu, nhan_dien, them_nguoi)
for f in frames:
    f.place(relx=0, rely=0, relheight=1, relwidth=1, anchor=NW)


def switch(frame):
    for f in frames:
        for widget in f.winfo_children():
            widget.destroy()
    if (frame == trang_chu):
        trangChu()
    elif (frame == nhan_dien):
        global run_detect
        run_detect = 1
        nhanDien()
    elif (frame == them_nguoi):
        reRenderImageButton()
        themNguoi()
    frame.tkraise()


def trangChu():
    f_trang_chu = tkinter.Frame(
        trang_chu, padx=0, pady=0, bg='lightblue')
    f_trang_chu.place(
        relx=0, rely=0, relheight=1, relwidth=1, anchor=NW)
    f_trang_chu.grid_columnconfigure(0, weight=1)
    f_trang_chu.grid_columnconfigure(1, weight=1)
    f_trang_chu.grid_columnconfigure(2, weight=1)
    f_trang_chu.grid_columnconfigure(3, weight=1)
    f_trang_chu.grid_columnconfigure(4, weight=1)
    f_trang_chu.grid_columnconfigure(5, weight=1)
    f_trang_chu.grid_rowconfigure(0, weight=1)
    f_trang_chu.grid_rowconfigure(1, weight=1)

    tkinter.Label(f_trang_chu, image=bg_image, anchor=W).place(
        relx=0, rely=0, relheight=1, relwidth=1, anchor=NW)

    tkinter.Label(f_trang_chu, text="Chọn chức năng", font=font_header1, bg='#1CA7E4', fg="white").grid(
        column=0, row=0, columnspan=6)
    tkinter.Button(f_trang_chu, text="Nhận diện", font=font_header2,  bg="#1AAAEA", fg="white", command=lambda: switch(
        nhan_dien)).grid(column=2, row=1, columnspan=1, sticky=N)
    tkinter.Button(f_trang_chu, text="Thêm người", font=font_header2, bg="#1AAAEA", fg="white", command=lambda: switch(
        them_nguoi)).grid(column=3, row=1, columnspan=1, sticky=N)

person_pre = 'Unknown Person'
img_person = Image.open('Face_image/Person/'+person_pre+'.jpg')
img_person = img_person.resize((200, 200), Image.ANTIALIAS)
img_ = ImageTk.PhotoImage(img_person)

def nhanDien():
    f_nhan_dien = tkinter.Frame(nhan_dien)
    f_nhan_dien.place(
        relx=0, rely=0, relheight=1, relwidth=1, anchor=NW)
    f_nhan_dien.grid_columnconfigure(0, weight=1)
    # f_nhan_dien.grid_columnconfigure(1, weight=1)
    f_nhan_dien_left = tkinter.Frame(
        f_nhan_dien, bg=lightGray, padx=20, pady=5)
    f_nhan_dien_left.place(
        relx=0, rely=0, relheight=1, relwidth=0.6, anchor=NW)

    f_nhan_dien_right = tkinter.Frame(f_nhan_dien, bg=white, padx=30, pady=5)
    f_nhan_dien_right.place(
        relx=1, rely=0, relheight=1, relwidth=0.4, anchor=NE)

    f_nhan_dien_left.grid_columnconfigure(0, weight=1)
    f_nhan_dien_left.grid_columnconfigure(1, weight=1)
    f_nhan_dien_left.grid_rowconfigure(0, weight=1)
    f_nhan_dien_left.grid_rowconfigure(1, weight=1)
    f_nhan_dien_left.grid_rowconfigure(2, weight=1)
    f_nhan_dien_left.grid_rowconfigure(3, weight=9)
    f_nhan_dien_left.grid_rowconfigure(4, weight=3)
    # f_nhan_dien_right.grid_columnconfigure(0, weight=1)
    # f_nhan_dien_right.grid_columnconfigure(1, weight=4)
    # f_nhan_dien_right.grid_columnconfigure(2, weight=1)
    # f_nhan_dien_right.grid_rowconfigure(0, weight=1)
    # f_nhan_dien_right.grid_rowconfigure(1, weight=1)
    # f_nhan_dien_right.grid_rowconfigure(2, weight=1)
    # f_nhan_dien_left.grid_rowconfigure(4, weight=5)

    tkinter.Button(f_nhan_dien_left, text="Trở về", font=font_content, command=lambda: switch(
        trang_chu)).grid(column=0, row=0, columnspan=2, sticky=NW)
    tkinter.Label(f_nhan_dien_left, text="Nhận diện khuôn mặt",
                  font=font_header1, anchor=W).grid(column=0, row=1, columnspan=2, sticky=W)
    tkinter.Label(f_nhan_dien_left,
                  text="Đưa mặt vào trước camera để nhận diện",
                  font=font_content, anchor=W, wraplength=500, justify=LEFT).grid(row=2, column=0, columnspan=2, sticky=NW)
    camera = tkinter.Label(f_nhan_dien_left, text="", image=default_them_nguoi)
    camera.grid(column=0, row=3, columnspan=2, sticky=NW)
    tkinter.Button(f_nhan_dien_left, text="Bắt đầu nhận diện", font=font_header2, bg="blue", fg="white",
                   
                   command=lambda: start_nhandien()).grid(column=0, row=4, columnspan=2, sticky=N, padx=5, pady=5)
    img_label = tkinter.Label(f_nhan_dien_right, image=img_)
    img_label.place(relx=0.5, rely=0.3,  anchor=N)
    name_label = tkinter.Label(f_nhan_dien_right, text=person_pre, font=font_header1, anchor=W)
    name_label.place(relx=0.5, rely=0.6, anchor=N)
    
    global run_detect, count_nhandien, face_dict
    run_detect = 1
    count_nhandien = -100
    face_dict = {}
    camera_nhandien(camera, name_label, img_label)

    def start_nhandien():
        global count_nhandien, face_dict
        count_nhandien = 2
        face_dict = {}

def takeAPhoto():
    global capture, flag
    flag = 0
    capture = 1

def themNguoi():
    f_them_nguoi = tkinter.Frame(them_nguoi)
    f_them_nguoi.place(
        relx=0, rely=0, relheight=1, relwidth=1, anchor=NW)

    f_them_nguoi_left = tkinter.Frame(
        f_them_nguoi, bg=lightGray, padx=20, pady=5)
    f_them_nguoi_left.place(
        relx=0, rely=0, relheight=1, relwidth=0.6, anchor=NW)

    f_them_nguoi_right = tkinter.Frame(f_them_nguoi, bg=white, padx=30, pady=5)
    f_them_nguoi_right.place(
        relx=1, rely=0, relheight=1, relwidth=0.4, anchor=NE)

    f_them_nguoi_left.grid_columnconfigure(0, weight=1)
    f_them_nguoi_left.grid_columnconfigure(1, weight=1)
    f_them_nguoi_left.grid_rowconfigure(0, weight=1)
    f_them_nguoi_left.grid_rowconfigure(1, weight=1)
    f_them_nguoi_left.grid_rowconfigure(2, weight=1)
    f_them_nguoi_left.grid_rowconfigure(3, weight=9)
    f_them_nguoi_left.grid_rowconfigure(4, weight=5)

    tkinter.Button(f_them_nguoi_left, text="Trở về", font=font_content, command=lambda: switch(
        trang_chu)).grid(column=0, row=0, columnspan=2, sticky=NW)
    tkinter.Label(f_them_nguoi_left, text="Nhận diện khuôn mặt",
                  font=font_header1, anchor=W).grid(column=0, row=1, columnspan=2, sticky=W)
    tkinter.Label(f_them_nguoi_left,
                  text="Để thêm một khuôn mặt mới, nhấn vào biểu tượng dấu cộng ở màn hình bên tay phải",
                  font=font_content, anchor=W, wraplength=500, justify=LEFT).grid(row=2, column=0, columnspan=2, sticky=NW)
    camera = tkinter.Label(f_them_nguoi_left, text="",
                           image=default_them_nguoi)
    camera.grid(
        column=0, row=3, columnspan=2, sticky=NW)

    captureButton = tkinter.Button(
        f_them_nguoi_left, text="Chụp ảnh", font=font_header2, command=takeAPhoto)
    captureButton.grid(column=0, row=4, columnspan=1, sticky=N)

    finishButton = tkinter.Button(
        f_them_nguoi_left, text="Kho Ảnh", font=font_header2, command=lambda: re_train(camera))
    finishButton.grid(column=1, row=4, columnspan=1, sticky=N)

    # RIGHT
    f_them_nguoi_right.grid_columnconfigure(0, weight=1)
    f_them_nguoi_right.grid_columnconfigure(1, weight=1)
    f_them_nguoi_right.grid_columnconfigure(2, weight=1)
    f_them_nguoi_right.grid_rowconfigure(0, weight=2)
    f_them_nguoi_right.grid_rowconfigure(1, weight=2)
    f_them_nguoi_right.grid_rowconfigure(2, weight=2)
    f_them_nguoi_right.grid_rowconfigure(3, weight=3)

    tkinter.Label(f_them_nguoi_right, text="Thêm khuôn mặt mới", bg=white,
                  font=font_header2, fg='lightblue', justify=CENTER).grid(column=0, row=0, columnspan=3, sticky=S)
    tkinter.Label(f_them_nguoi_right, image=arow, bg=white, justify=CENTER).grid(
        column=0, row=1, columnspan=3, sticky=W)

    tkinter.Button(f_them_nguoi_right, image=button_them_nguoi, relief=FLAT, command=lambda: getName(camera)).grid(
        column=0, row=2, columnspan=1, sticky=NW)
    global capture
    capture = - 20
    listButton = []
    for i in range(5):
        listButton.append(tkinter.Button(f_them_nguoi_right,
                          image=listImage[i], relief=FLAT))
    '''Set image for button'''

    listButton[0].grid(column=1, row=2, columnspan=1, sticky=N)
    listButton[1].grid(column=2, row=2, columnspan=1, sticky=NE)
    listButton[2].grid(column=0, row=2, columnspan=1, sticky=SW)
    listButton[3].grid(column=1, row=2, columnspan=1, sticky=S)
    listButton[4].grid(column=2, row=2, columnspan=1, sticky=SE)


listImage = [default_empty, default_empty,
             default_empty, default_empty, default_empty]


def reRenderImageButton():
    base_path_image = "Person_image/"
    folder_image = [os.path.join(base_path_image, f)
                    for f in os.listdir(base_path_image)]
    len_i = len(folder_image)
    i = 1
    thred = 6
    while i < thred:
        try:
            path_folder = folder_image[len_i-i]
            image_path = path_folder
            load_img = (Image.open(
                image_path))
            load_img = load_img.resize(
                (60, 60), Image.ANTIALIAS)
            listImage[i-1] = ImageTk.PhotoImage(load_img)
            i += 1
        except:
            path_folder = folder_image[len_i - i - 6]
            image_path = path_folder
            load_img = (Image.open(
                image_path))
            load_img = load_img.resize(
                (60, 60), Image.ANTIALIAS)
            listImage[i-1] = ImageTk.PhotoImage(load_img)
            i += 1
            

'''CAMERA'''
cap = cv2.VideoCapture(0)
start = False
name = ""
count = 0
flag = 0

# Define function to show frame

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
devide = "cpu"
print(device)

from processing_face import transform_face
from deepface import DeepFace

def show_frames(camera):
    global capture, count, name, flag
    if start == False:
        return
    # Get the latest frame and convert into Image
    frame = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)
    # cap window
    if capture > 0:
        # tạo tên ảnh
            try:
                frame, face, x, y = Mtcnn_face_detect(frame)
                list_face = transform_face(face)
                i = 1
                for face in list_face:
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    cv2.imwrite('Person_image/' + name + '_' + str(i) + '.jpg', face)
                    i+=1
                capture -= 1
            except Exception as e:
                print(e)
        # plt.imshow(photoSave, cmap='gray')
        # get img capture
        # end capture
    # Repeat after an interval to capture continiously
    img = Image.fromarray(frame)
    img = img.resize((560, int(3*560/4)), Image.ANTIALIAS)

    # Convert image to PhotoImage
    imgtk = ImageTk.PhotoImage(img)
    camera.imgtk = imgtk
    camera.configure(image=imgtk)
    if capture <= 0 and capture >= -10 and flag == 1:
        capture = -100
        os.remove('Person_image/representations_facenet.pkl')
        # tmp = DeepFace.find(img_path = "tmp.jpg", db_path = "Person_image", model_name = "Facenet", enforce_detection = 'False')
        messagebox.showinfo("Thông báo", "Thêm người thành công!!")
        switch(trang_chu)        
        return
    if capture <= 0 and capture >= -10:
        flag = 1
    camera.after(5, lambda: show_frames(camera))

def camera_nhandien(camera, name_label, img_label):
    global count_nhandien, face_dict
    # Get the latest frame and convert into Image
    frame = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB) # Lấy frame từ camera
    if count_nhandien > 0:  # Kiểm tra xem chụp đủ 10 ảnh chưa
        frame, id, x, y = Mtcnn_face_predict(frame)
        if id is not None:
            try:
                cv2.putText(frame, str(round((11-count_nhandien+1) / 11 * 100, 2))+'%',
                            (x+20, y-20), cv2.FONT_HERSHEY_DUPLEX, 1, (239, 50, 239), 2, cv2.LINE_AA)   # In ra số ảnh đã chụp
                # Đếm số lần được dự đoán của các nhãn
                if id in face_dict.keys():
                    face_dict[id] += 1
                    count_nhandien -= 1
                else:
                    face_dict[id] = 1
                    count_nhandien -= 1
            except Exception as e:
                print(e)

    img = Image.fromarray(frame)
    img = img.resize((560, int(3*560/4)), Image.ANTIALIAS)
    # Convert image to PhotoImage
    imgtk = ImageTk.PhotoImage(img)
    camera.imgtk = imgtk
    camera.configure(image=imgtk)
    if count_nhandien <= 0 and count_nhandien >= -10:
        count_nhandien = -100
        if "Giả Mạo" in face_dict.keys():
            name = "Giả Mạo"
        else:
            max = 0
            name = ''
            for i in face_dict.keys():
                if face_dict[i] > max:
                    max = face_dict[i]
                    name = i
        try:
            img_person = Image.open(base_dir + '/Face_image/Person/'+name+'.jpg')
        except:
            img_person = Image.open(base_dir + '/Face_image/Person/Unknown Person.jpg')
        img_person = img_person.resize((200, 200), Image.ANTIALIAS)
        img_person = ImageTk.PhotoImage(img_person)
        name_label['text'] = name
        img_label['image'] = img_person
        question = messagebox.askquestion("KẾT QUẢ NHẬN DIỆN", 'Đây có phải là?\n' + name)
    camera.after(5, lambda: camera_nhandien(camera, name_label, img_label))


def startVideo(camera):
    global start
    start = True
    show_frames(camera)


def re_train(camera):
    global start
    camera['image'] = default_them_nguoi
    # start = False
    # switch(trang_chu)
    os.system(r'start Person_image')
    

def save(file_name, img, path):
    # Set vị trí lưu ảnh
    os.chdir(path)
    # Lưu ảnh
    cv2.imwrite(file_name, img)


def getName(camera):
    top = tkinter.Toplevel(win)

    top.title("window")
    top.geometry("230x100")

    label = tkinter.Label(top, text="Nhập tên:", font=font_header1)
    label.place(relx=0.5, rely=0.2, anchor=N)

    text = tkinter.Text(top, height=1, width=20)
    text.place(relx=0.5, rely=0.5, anchor=N)

    def get():
        global count, name
        name = text.get(1.0, END)[0:-1]
        startVideo(camera)
        top.destroy()

    button = tkinter.Button(top, text="OK", command=get)
    button.place(relx=0.5, rely=0.8, anchor=N)


switch(trang_chu)
win.mainloop()