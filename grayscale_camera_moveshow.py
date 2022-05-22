"統計每(幾)秒判斷的結果，輸出出現最多次的情緒作為判斷結果"
from keras.models import load_model
model = load_model('65%_3000_64X64_FER_64-model')#左上設路徑 & 輸入模型名稱

from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import cv2
import os
import time
from PIL import Image
#%%

def detectFace(img):

    img = cv2.imread(img) # 讀取圖檔
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 透過轉換函式轉為灰階影像
    color = (0, 0, 0)  # 定義框的顏色
    
    # OpenCV 人臉識別分類器
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    # 調用偵測識別人臉函式
    faceRects = face_classifier.detectMultiScale(
        grayImg, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    
    # 大於 0 則檢測到人臉
    if len(faceRects):  
        # 框出每一張人臉
        for faceRect in faceRects: 
            x, y, w, h = faceRect
            #cv2.rectangle(img, (x, y), (x + h, y + w), color, 0)
            crop_img = img[y-25:y+w+25, x-25:x+h+25]
            #crop_img = img[y:y+w, x:x+h]
    
    # 將結果圖片輸出
    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('crop.jpg', crop_img)

#%%記錄各時段情緒
star_time = 0
time_emotion = []
#%%開啟攝影機
'''
cap = cv2.VideoCapture(0)#開啟鏡頭'''
#%%開起影片
cap = cv2.VideoCapture('E:\\學校課程\\video_data\\test2.mp4')
#%%fps
fps = cap.get(cv2.CAP_PROP_FPS)
#%%錄影保存
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('C:\\Users\\MSI\\Desktop\\作業\\專題\\CNN_face_recognition\\data_grayscale_camera_moveshow\\output.mp4', fourcc, fps, (640, 480))
#%%

#統計每(幾)秒判斷的結果
seconds = time.time()
emotion_names = {"angry":0, "fear":0, "happy":0, "neutral":0, "sad":0, "surprise":0}
n=0
number=0
fps_n = 0
while(True):
    
    number+=1
    
    ret,frame = cap.read()#捕獲一幀影象
    
    
    cv2.imwrite("_face"+str(number)+".jpg", frame)
    
    #裁切臉部
    try:
        detectFace("_face"+str(number)+".jpg")
    
        image = cv2.imread("crop.jpg")
        imgs = cv2.resize(image, (64, 64))#跟建模時的input_shape需相同
        cv2.imwrite('crop.jpg', imgs)
        
        #情緒辨識
        IMAGE_PATH="./crop.jpg" #輸入圖片
    
        img=tf.keras.preprocessing.image.load_img(IMAGE_PATH,
                                                  grayscale=False,
                                                  color_mode='grayscale',
                                                  target_size=(64, 64, 1))#跟建模時的input_shape需相同
        img=tf.keras.preprocessing.image.img_to_array(img)
        plt.imshow(img/255.)
        predictions=model.predict(np.array([img]))
        #print(predictions)
#%%
        class_names = ["angry","fear","happy","neutral","sad","surprise"]
        m=0
        for i in predictions:
            for j in i:
                emotion_names[class_names[m]]+=j
                m+=1
        #print(class_names[m])

        color = (0, 0, 0)  # 定義框的顏色
        # OpenCV 人臉識別分類器
        face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        
        # 調用偵測識別人臉函式
        faceRects = face_classifier.detectMultiScale(
            frame, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        
        text = max(emotion_names, key=lambda key: emotion_names[key])
        # 大於 0 則檢測到人臉
        if len(faceRects):  
            # 框出每一張人臉
            for faceRect in faceRects: 
                x, y, w, h = faceRect
                cv2.rectangle(frame, (x, y), (x + h, y + w), color, 0)
                cv2.putText(frame, text, (x + h, y + w), cv2.FONT_HERSHEY_TRIPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                break
            
        out.write(frame)#保存影象
        cv2.imshow('frame',frame)
#%%
        fps_n += 1
        n+=1
        seconds2 = time.time()
        
        #print(int(seconds2)-int(seconds))
        if (n == fps):
            
            for i in emotion_names:
                emotion_names[i] = emotion_names[i]/n
                
            max_emotion_class = max(emotion_names, key=lambda key: emotion_names[key])
            max_emotion = emotion_names[max_emotion_class]
            max2_emotion_class = ""
            max2_emotion = 0
            
            for i in emotion_names:
                if emotion_names[i] > max2_emotion and emotion_names[i] < emotion_names[max_emotion_class]:
                    max2_emotion_class = i
                    max2_emotion = emotion_names[max2_emotion_class]

            print((emotion_names))
            max_emotion_class = max(emotion_names, key=lambda key: emotion_names[key])
            #print(n)
            print("預測結果1:",max_emotion_class,emotion_names[max_emotion_class]*100,"%")
            print("預測結果2:",max2_emotion_class,max2_emotion*100,"%")
            
            #顯示測試表情
            test_image = Image.open('crop.jpg')
            plt.imshow(test_image)
            plt.show()
            
            #%%紀錄這時段情緒
            time_emotion.append([star_time,fps_n,max_emotion_class])
            star_time = fps_n
             #%%
            emotion_names = {"angry":0, "fear":0, "happy":0, "neutral":0, "sad":0, "surprise":0}
            n=0

            #time.sleep(1)   
    except:
        #print("沒有偵測到臉")
        cv2.putText(frame, "No Face", (200,250), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 255), 1, cv2.LINE_AA)
        
        out.write(frame)#保存影象
        cv2.imshow('frame',frame)
        #time.sleep(1)

#%%
    #刪除檔案
    fileTest = r"C:/Users/MSI/Desktop/作業/專題/CNN_face_recognition/_face"+str(number)+".jpg"
    try:
        os.remove(fileTest)
    except OSError as e:
        print(e)
    else:
        pass
        #print("File is deleted successfully")
        
    #判斷按鍵，如果按鍵為q，退出迴圈
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite("_face"+str(number)+".jpg", frame)
        
        for i in emotion_names:
            emotion_names[i] = emotion_names[i]/n
                
        max_emotion_class = max(emotion_names, key=lambda key: emotion_names[key])
        max_emotion = emotion_names[max_emotion_class]
        max2_emotion_class = ""
        max2_emotion = 0
        
        for i in emotion_names:
            if emotion_names[i] > max2_emotion and emotion_names[i] < emotion_names[max_emotion_class]:
                max2_emotion_class = i
                max2_emotion = emotion_names[max2_emotion_class]

        print((emotion_names))
        max_emotion_class = max(emotion_names, key=lambda key: emotion_names[key])
        #print(n)
        print("預測結果1:",max_emotion_class,emotion_names[max_emotion_class]*100,"%")
        print("預測結果2:",max2_emotion_class,max2_emotion*100,"%")
        
        #顯示測試表情
        test_image = Image.open('crop.jpg')
        plt.imshow(test_image)
        plt.show()
        
        #%%紀錄這時段情緒
        time_emotion.append([star_time,fps_n,max_emotion_class])
        star_time = fps_n
         #%%
        emotion_names = {"angry":0, "fear":0, "happy":0, "neutral":0, "sad":0, "surprise":0}
        n=0

            #time.sleep(1)
        break
#%%
number=0
end_time = time.time()
print("時長:",fps_n/fps)
print("時段:",time_emotion)
#%%整理各段情緒
emotion_percent = {"angry":0, "fear":0, "happy":0, "neutral":0, "sad":0, "surprise":0}
s = time_emotion[0][2]
t = 0
T = []
o=0
for i in time_emotion:
    emotion_percent[i[2]] += (i[1]-i[0])/fps_n*100
    #%%整理
    o+=1
    e = i[2]
    if e != s:
        T.append([t,i[0],s])
        s = e
        t=i[0]
        
    if o == len(time_emotion):

        T.append([t,i[1],s])
        #%%
print("整理:",T)
print("情緒占比:",emotion_percent)
#%%
cap.release()#關閉相機
out.release()
cv2.destroyAllWindows()#關閉視窗