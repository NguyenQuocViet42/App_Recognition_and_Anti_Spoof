from facenet_pytorch import MTCNN
import torch
import cv2
from deepface import DeepFace
import albumentations as A
import numpy as np

import os
import sys
# sys.path.append(os.getcwd())
sys.path.append(os.getcwd()+'/predicter/')
# sys.path.append(os.getcwd()+'/Person_image')
import model.Face_Fake_Net as Face_Fake_Net

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("\nLoading Deep Face\n\n")
tmp = DeepFace.find(img_path = "tmp.jpg", db_path = "Person_image", model_name = "Facenet", enforce_detection = 'False')

def Deep_face_predict(face):
    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    except:
        return None
    cv2.imwrite('tmp.jpg', face)
    result = DeepFace.find(img_path = "tmp.jpg", db_path = "Person_image", model_name = "Facenet", enforce_detection = 'False')
    try:
        name = result.iloc[0][0].split('/')[-1]
    except:
        return "Unknown Person"
    name = name.split('_')[0]
    return name

print("\nLoading MTCNN\n\n")

mtcnn = MTCNN(margin=20, keep_all=False, select_largest=True,
              post_process=False, device=device)

# with open("Person_image\\representations_facenet.pkl", "rb") as f:
#         mtcnn = pickle.load(f)

def Mtcnn_face_detect(frame):
    faces, _ = mtcnn.detect(frame, landmarks=False)
    if faces is None:
        return None, None, None, None
    for face_info in faces:
        # Lấy thông tin về tọa độ hình chữ nhật bao quanh khuôn mặt
        x, y, width, height = face_info
        x, y, width, height = int(x), int(y), int(width), int(height)
        a = (x + width) // 2
        b = (y + height) // 2
        r = max((width - x), (height - y)) // 2 + 15
        x_1 = a - r
        y_1 = b - r
        x_2 = a + r
        y_2 = b + r
        # Vẽ hình chữ nhật xung quanh khuôn mặt
        cv2.rectangle(frame, (x_1, y_1), (x_2, y_2), (0, 255, 0), 2)
        
        face = frame[y_1:y_2, x_1:x_2]
        return frame, face, x, y
    
def Mtcnn_face_predict(frame):
    faces, _ = mtcnn.detect(frame, landmarks=False)
    if faces is None:
        return None, None, None, None
    for face_info in faces:
        # Lấy thông tin về tọa độ hình chữ nhật bao quanh khuôn mặt
        x, y, width, height = face_info
        x, y, width, height = int(x), int(y), int(width), int(height)
        a = (x + width) // 2
        b = (y + height) // 2
        r = max((width - x), (height - y)) // 2
        x_1 = a - r
        y_1 = b - r
        x_2 = a + r
        y_2 = b + r
        
        # Vẽ hình chữ nhật xung quanh khuôn mặt
        
        # print(x_1, y_1, x_2, y_2)
        
        # try:
        #     face_1 = frame[y_1 : y_2, x_1 : x_2]
        # except Exception as e:
        #     face_1 = frame[y: height, x : width]
        #     print(e)
            
        # try: 
        #     face_2 = frame[y_1 - 5 : y_2 + 5, x_1 - 5 : x_2 + 5]
        # except Exception as e:
        #     face_2 = frame[y: height, x : width]
        #     print(e)
        
        try:
            face_3 = frame[y_1 - 15 : y_2 + 15, x_1 - 15 : x_2 + 15]
        except Exception as e:
            face_3 = frame[y: height, x : width]
            print(e)
        
        try:
            face_4 = frame[y_1 - 25: y_2 + 25, x_1 - 25: x_2 + 25]
        except Exception as e:
            face_4 = frame[y: height, x : width]
            print(e)
            
        try:
            face_5 = frame[y_1 - 35 : y_2 + 35, x_1 - 35 : x_2 + 35]
            cv2.imwrite('test.jpg', face_5)
        except Exception as e:
            face_5 = frame[y: height, x : width]
            print(e)
            
        # list_face = [face_1, face_2, face_3, face_4, face_5]
        list_face = [face_3, face_4, face_5]
        # Check Spoof
        count = 0
        for f in list_face:
            if Face_face_net_predict(f):
                count += 1
        
        if count >= 2:
            face = frame[b - (r+15) : b + (r+15), a - (r+15) : a + (r+15)]
            id = Deep_face_predict(face)
        else:
            id = "Giả Mạo"
        
        cv2.rectangle(frame, (x_1, y_1), (x_2, y_2), (0, 255, 0), 2)
        return frame, id, x, y
    
normalize_transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print("\nLoading Face Fake Net\n\n")

net = Face_Fake_Net.Face_Fake_Net()
# test_check_point/Epoch_28_GPU_1_best_check_point.pth 95.5
tmp = torch.load('predicter/check_point/Epoch_58_GPU_1_best_check_point.pth', map_location='cpu')
# tmp = torch.load('/media/quocviet/New Volume/Comit/FAS/data/Face_Anti_Spoofing_Face_Fake_Net/check_point/0_GPU_200K_test_best_check_point.pth', map_location=lambda storage, loc: storage)
corrected_dict = {}
def fix_key(wrong_key):
    return wrong_key.replace('module.', "")
for wrong_key, value in tmp.items():
    correct_key = fix_key(wrong_key)  # Replace fix_key with your function to fix keys
    corrected_dict[correct_key] = value
net.load_state_dict(corrected_dict)
net.to(device);
net.eval();

def Face_face_net_predict(image):
    image = cv2.resize(image,(224,224))
    image = normalize_transform(image = image)['image']
    image = image.astype(np.float32)
    image = torch.tensor(image)
    image = image.permute(2, 0, 1).unsqueeze(0)
    image = image.to(device)
    prob = net(image)
    result = torch.argmax(torch.softmax(prob, dim = 1)).item() == 0
    # print(torch.max(torch.softmax(prob, dim = 1)).item())
    if result == True:
        if torch.max(torch.softmax(prob, dim = 1)).item() > 0.93:
            return True
        else:
            return False
    else: 
        return False