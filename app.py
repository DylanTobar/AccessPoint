
import http
import os
import re
import time
from tkinter import Button
from urllib import request
import cv2
from grpc import server
import imutils
import numpy as np
#import firebase_admin
#from firebase_admin import credentials, storage
from flask import Flask, Response, render_template
#import RPi.GPIO as GPIO
#from mfrc522 import SimpleMFRC522

#prueba de firebase
#cred = credentials.Certificate("./key.json")
#app = firebase_admin.initialize_app(cred, {'storageBucket':'proyectograduacion-435b1.appspot.com'})
#bucket = storage.bucket()
#blob= bucket.get_blob ('Data')

app = Flask(__name__)
dataPath = 'C:/Users/dylan/Desktop/Tesis/facial_recognition/Data' 
#dataPath = 'Data' 
imagePaths = os.listdir(dataPath)

#RELAY = 17
#GPIO.setwarnings(False)
#GPIO.setmode(GPIO.BCM)
#GPIO.setup(RELAY, GPIO.OUT)
#GPIO.output(RELAY,GPIO.LOW)


#def generate_rfid():
     #reader = SimpleMFRC522()
     #try:
     #    id.text = reader.read()
     #finally:
     #    GPIO.cleanup()

def generate_rec():
     face_recognizer = cv2.face.LBPHFaceRecognizer_create()
     face_recognizer.read('modeloLBPHFace.xml')
     #cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)     
     cap = cv2.VideoCapture(0)
     face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
     #prevTime = 0
     #doorUnlock = False
     while True:
          ret, frame = cap.read()
          if ret:
               gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
               auxFrame = gray.copy()
               faces = face_detector.detectMultiScale(gray, 1.3, 5)
               for (x, y, w, h) in faces:
                    rostro = auxFrame[y:y+h,x:x+w]
                    rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
                    result = face_recognizer.predict(rostro)
                    if result[1] < 70:  
                         cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
                         cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
                         GPIO.output(RELAY,GPIO.HIGH)
                         #prevTime = time.time()
                         #doorUnlock = True
                         #print("door unlock")

                    else:
                         cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
                         cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
               (flag, encodedImage) = cv2.imencode(".jpg", frame)
                           
               if not flag:
                    continue
               yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                    bytearray(encodedImage) + b'\r\n')
               #if doorUnlock == True and time.time() & prevTime > 5:
               #     doorUnlock = False
               #     GPIO.output(RELAY,GPIO.LOW)
               #     print("door lock")   

          
def generate_reg():
     personName = "Dylan"
     personPath = dataPath + '/' + personName
     if not os.path.exists(personPath):
          print('Carpeta creada: ',personPath)
          os.makedirs(personPath)

     #cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)     
     cap = cv2.VideoCapture(0)     
     faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
     count = 0
     
     while True:

          ret, frame = cap.read()
          if ret == False: break
          frame =  imutils.resize(frame, width=640)
          gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
          auxFrame = frame.copy()

          faces = faceClassif.detectMultiScale(gray,1.3,5)

          for (x,y,w,h) in faces:
               cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
               rostro = auxFrame[y:y+h,x:x+w]
               rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
               cv2.imwrite(personPath + '/rotro_{}.jpg'.format(count),rostro)
               count = count + 1
          cv2.imshow('frame',frame)

          k =  cv2.waitKey(1)
          if k == 27 or count >= 20:  #cantidad de fotos
               break
    # cv2.destroyAllWindows()

     peopleList = os.listdir(dataPath)
     print('Lista de personas: ' ,peopleList)

     labels = []
     facesData = []
     label = 0

     for nameDir in peopleList:
          personPath = dataPath + '/' + nameDir
          print('Leyendo las imágenes')

          for fileName in os.listdir(personPath):
               print('Rostros: ', nameDir + '/' + fileName)
               labels.append(label)
               facesData.append(cv2.imread(personPath+'/'+fileName,0))
          label = label + 1

     face_recognizer = cv2.face.LBPHFaceRecognizer_create()

     print("Entrenando...")
     face_recognizer.train(facesData, np.array(labels))

     face_recognizer.write('modeloLBPHFace.xml')
     print("Modelo almacenado...")

#@app.route("/rfid")
#def rfid():
#     return Response(generate_rfid())
registrar = Button
@app.route("/")
def index():
     return render_template("index.html")

@app.route("/reg_rostro")
def registrar_rostro():
     return render_template("registrar.html")

@app.route("/video_feed")
def video_feed():
     return Response(generate_rec(),
          mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route("/video_feed2")
def video_feed2():
     return Response(generate_reg(),
          mimetype = "multipart/x-mixed-replace; boundary=frame")
          
@app.route("/visitas")
def visitas():
     return render_template("visitas.html")
@app.route("/huellas")
def huellas():
     return render_template("huellas.html")
@app.route("/aceptado")
def aceptado():
     return render_template("aceptado.html")
@app.route("/empleados")
def empleados():
     return render_template("empleados.html")


if __name__ == "__main__":
     app.run(debug=True)

