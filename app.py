import webbrowser
from flask import Flask
from flask import render_template
from flask import Response
import cv2
import os
from flask import Flask
import imutils
import numpy as np



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
cap.release()