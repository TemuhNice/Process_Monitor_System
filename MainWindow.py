# -*- coding: utf-8 -*-
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QTimer, QTime
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMessageBox, QDialog
import numpy as np
from PyQt5.QtWidgets import QWidget
import psutil
#import RPi.GPIO as GPIO
import time
import pygame
#from pyembedded.raspberry_pi_tools.raspberrypi import PI
import psutil
import datetime

#GPIO.setmode(GPIO.BCM)
#GPIO.setup(23,GPIO.IN)
#GPIO.setup(27,GPIO.IN)
#GPIO.setup(26,GPIO.IN)
pygame.mixer.init()

whT = 320
confThreshold = 0.5
nmsThreshold = 0.3

classesFile = 'coco.names'
classNames = []
with open(classesFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
modelConfiguration = 'yolov3-tiny.cfg'
modelWeights = 'yolov3-tiny.weights'
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def findObjects(outputs,img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w,h = int(det[2]*wT), int(det[3]*hT)
                x,y = int(det[0]*wT - w/2), int(det[1]*hT - h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold)
    presence = 0
    for i in indices:
        i = i[0]
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        label = 'person'
        if label == str(classNames[classIds[i]]):
            cv2.rectangle(img,(x,y),(x+w, y+h),(0,255,0),2)
            cv2.putText(img,f'{classNames[classIds[i]].upper()} ',
            (x,y-10), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
            presence = 1
        else:
            continue
    return presence


def returnCameraIndexes():
    index = 0
    arr = []
    i = 0
    while i < 3:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            arr.append(index)
            cap.release()
        index += 1
        i += 1
    return arr




class Ui_MainScreen(object):
    def setupUi(self, MainScreen):
        MainScreen.setObjectName("MainScreen")
        MainScreen.setWindowModality(QtCore.Qt.NonModal)
        MainScreen.resize(900, 597)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainScreen.sizePolicy().hasHeightForWidth())
        MainScreen.setSizePolicy(sizePolicy)
        MainScreen.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainScreen.setWindowIcon(icon)
        MainScreen.setWindowOpacity(1.0)
        MainScreen.setAutoFillBackground(True)
        MainScreen.setStyleSheet("")
        MainScreen.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        MainScreen.setAnimated(False)
        MainScreen.setDockNestingEnabled(False)
        MainScreen.setUnifiedTitleAndToolBarOnMac(False)
        self.centralwidget = QtWidgets.QWidget(MainScreen)
        self.centralwidget.setObjectName("centralwidget")
        self.vidCapture = QtWidgets.QLabel(self.centralwidget)
        self.vidCapture.setGeometry(QtCore.QRect(20, 20, 640, 480))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.vidCapture.sizePolicy().hasHeightForWidth())
        self.vidCapture.setSizePolicy(sizePolicy)
        self.vidCapture.setStyleSheet("")
        self.vidCapture.setFrameShape(QtWidgets.QFrame.Box)
        self.vidCapture.setFrameShadow(QtWidgets.QFrame.Raised)
        self.vidCapture.setLineWidth(2)
        self.vidCapture.setText("")
        self.vidCapture.setObjectName("vidCapture")
        self.cameras = QtWidgets.QComboBox(self.centralwidget)
        self.cameras.setGeometry(QtCore.QRect(670, 80, 221, 31))
        self.cameras.setStyleSheet("QComboBox::drop-down \n"
"{\n"
"    width: 0px;\n"
"    height: 0px;\n"
"    border: 0px;\n"
"}\n"
"\n"
"QComboBox QAbstractItemView {\n"
"  color: rgb(85, 170, 255);    \n"
"  background-color: #373e4e;\n"
"  padding: 10px;\n"
"  selection-background-color: rgb(39, 44, 54);\n"
"  font: 9pt ""Segoe UI"";\n"
"}")
        self.cameras.setFrame(True)
        self.cameras.setObjectName("cameras")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(670, 40, 211, 41))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setAutoFillBackground(False)
        self.label.setStyleSheet("color: rgb(255, 255, 255);")
        self.label.setTextFormat(QtCore.Qt.RichText)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setWordWrap(False)
        self.label.setObjectName("label")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(670, 330, 221, 161))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        self.textBrowser.setFont(font)
        self.textBrowser.setStyleSheet("background-color: rgba(255, 255, 255, 0);\n"
"color: rgb(255, 255, 255);")
        self.textBrowser.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.textBrowser.setFrameShadow(QtWidgets.QFrame.Raised)
        self.textBrowser.setObjectName("textBrowser")
        self.lcdNumber = QtWidgets.QLCDNumber(self.centralwidget)
        self.lcdNumber.setGeometry(QtCore.QRect(770, 0, 131, 31))
        self.lcdNumber.setStyleSheet("color: rgb(255, 0, 0);\n"
"")
        self.lcdNumber.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.lcdNumber.setSegmentStyle(QtWidgets.QLCDNumber.Flat)
        self.lcdNumber.setProperty("intValue", 15)
        self.lcdNumber.setObjectName("lcdNumber")
        self.stopProcess = QtWidgets.QPushButton(self.centralwidget)
        self.stopProcess.setGeometry(QtCore.QRect(100, 520, 64, 60))
        self.stopProcess.setAutoFillBackground(False)
        self.stopProcess.setStyleSheet("")
        self.stopProcess.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/icons/stop.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.stopProcess.setIcon(icon1)
        self.stopProcess.setIconSize(QtCore.QSize(51, 51))
        self.stopProcess.setDefault(False)
        self.stopProcess.setFlat(True)
        self.stopProcess.setObjectName("stopProcess")
        self.buttonGroup = QtWidgets.QButtonGroup(MainScreen)
        self.buttonGroup.setObjectName("buttonGroup")
        self.buttonGroup.addButton(self.stopProcess)
        self.startProcess = QtWidgets.QPushButton(self.centralwidget)
        self.startProcess.setGeometry(QtCore.QRect(20, 520, 64, 60))
        self.startProcess.setAutoFillBackground(False)
        self.startProcess.setStyleSheet("")
        self.startProcess.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("start.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.startProcess.setIcon(icon2)
        self.startProcess.setIconSize(QtCore.QSize(51, 51))
        self.startProcess.setDefault(False)
        self.startProcess.setFlat(True)
        self.startProcess.setObjectName("startProcess")
        self.buttonGroup.addButton(self.startProcess)
        self.processProgress = QtWidgets.QProgressBar(self.centralwidget)
        self.processProgress.setGeometry(QtCore.QRect(180, 520, 711, 61))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.processProgress.setFont(font)
        self.processProgress.setStyleSheet(
"QProgressBar::chunk \n"
"{\n"
"background-color: #05B8CC;\n"
"} ")
        self.processProgress.setProperty("value", 0)
        self.processProgress.setAlignment(QtCore.Qt.AlignCenter)
        self.processProgress.setTextVisible(True)
        self.processProgress.setObjectName("processProgress")
        self.startVideo = QtWidgets.QPushButton(self.centralwidget)
        self.startVideo.setGeometry(QtCore.QRect(670, 117, 221, 31))
        self.startVideo.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(255, 32, 32);\n"
"")
        self.startVideo.setFlat(False)
        self.startVideo.setObjectName("startVideo")
        self.formLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.formLayoutWidget.setGeometry(QtCore.QRect(670, 160, 223, 131))
        self.formLayoutWidget.setObjectName("formLayoutWidget")
        self.formLayout = QtWidgets.QFormLayout(self.formLayoutWidget)
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.formLayout.setObjectName("formLayout")
        self.label_8 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_8.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.SpanningRole, self.label_8)
        self.label_2 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_2.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.smoke = QtWidgets.QLabel(self.formLayoutWidget)
        self.smoke.setStyleSheet("color: rgb(85, 255, 0);")
        self.smoke.setAlignment(QtCore.Qt.AlignCenter)
        self.smoke.setObjectName("smoke")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.smoke)
        self.label_4 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_4.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_4.setObjectName("label_4")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.flame = QtWidgets.QLabel(self.formLayoutWidget)
        self.flame.setStyleSheet("color: rgb(85, 255, 0);")
        self.flame.setAlignment(QtCore.Qt.AlignCenter)
        self.flame.setObjectName("flame")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.flame)
        self.label_6 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_6.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_6.setObjectName("label_6")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_6)
        self.motion = QtWidgets.QLabel(self.formLayoutWidget)
        self.motion.setStyleSheet("color: rgb(85, 255, 0);")
        self.motion.setAlignment(QtCore.Qt.AlignCenter)
        self.motion.setObjectName("motion")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.motion)
        self.label_12 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_12.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_12.setObjectName("label_12")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_12)
        self.cpuLoad = QtWidgets.QProgressBar(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        self.cpuLoad.setFont(font)
        self.cpuLoad.setStyleSheet("QProgressBar\n"
"{\n"
"color: black;\n"
"}\n"
"QProgressBar::chunk \n"
"{\n"
"background-color: #05B8CC;\n"
"} ")
        self.cpuLoad.setProperty("value", 24)
        self.cpuLoad.setAlignment(QtCore.Qt.AlignCenter)
        self.cpuLoad.setTextVisible(True)
        self.cpuLoad.setObjectName("cpuLoad")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.cpuLoad)
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(0, 0, 901, 601))
        self.label_9.setStyleSheet("background-image: url(:/bgs/Backg.jpg);")
        self.label_9.setText("")
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(680, 300, 211, 31))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_10.setFont(font)
        self.label_10.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_10.setObjectName("label_10")
        self.label_9.raise_()
        self.vidCapture.raise_()
        self.cameras.raise_()
        self.label.raise_()
        self.textBrowser.raise_()
        self.lcdNumber.raise_()
        self.stopProcess.raise_()
        self.startProcess.raise_()
        self.processProgress.raise_()
        self.startVideo.raise_()
        self.formLayoutWidget.raise_()
        self.label_10.raise_()
        MainScreen.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainScreen)
        QtCore.QMetaObject.connectSlotsByName(MainScreen)
        self.lcdNumber.setDigitCount(8)
        #timer = QTimer()
        #timer.timeout.connect(self.displayTime)
        self.loadCameras()
        self.startVideo.clicked.connect(lambda: self.clicked())
        self.startProcess.clicked.connect(lambda: self.startClicked())
        self.stopProcess.clicked.connect(lambda: self.stopClicked())
        self.Camera1Thread = Camera1Thread()
        self.Camera2Thread = Camera2Thread()
        self.Camera3Thread = Camera3Thread()
        self.cpu = CpuLoad()
        self.cpu.start()
        self.cpu.signal.connect(self.cpuUsage)
        self.cpu.flameSig.connect(self.labelUpd1)
        self.cpu.motionSig.connect(self.labelUpd2)
        self.cpu.smokeSig.connect(self.labelUpd3)
        self.processHandle = ProcessRun()
        self.processStarted = False
        self.lastProgressState = 0
        self.processHandle.textbrowsersignal.connect(self.textBrowsUpd)
        self.Fire = 0
        self.Smoke = 0
        self.MotionSensor = 0
        self.isSomebodyHere = 0
        self.camActive = 0
        self.prevCam = 4
        self.isRunFirstTime = 1        
        
        
        
    def clicked(self):
        if self.isRunFirstTime:
            if len(self.cameras) > 0:
                self.isRunFirstTime = 0
                self.prevCam = self.cameras.currentIndex()
                self.camActive = 1
                if self.cameras.currentIndex() == 0:
                    self.Camera1Thread.start()
                    self.Camera1Thread.ImageUpdate.connect(self.updImage)
                if self.cameras.currentIndex() == 1:
                    self.Camera2Thread.start()
                    self.Camera2Thread.ImageUpdate.connect(self.updImage)
                if self.cameras.currentIndex() == 2:
                    self.Camera3Thread.start()
                    self.Camera3Thread.ImageUpdate.connect(self.updImage)
            else:
                error = QMessageBox()
                error.setWindowTitle("ОШИБКА")
                error.setText("Камеры не найдены")
                error.setIcon(QMessageBox.Warning)
                error.setStandardButtons(QMessageBox.Ok)
                error.exec_()
        else:
            if self.prevCam == self.cameras.currentIndex():
                if self.camActive == 1:
                    self.camActive = 0
                    if self.prevCam == 0:
                        self.Camera1Thread.stop()
                    if self.prevCam == 1:
                        self.Camera2Thread.stop()
                    if self.prevCam == 2:
                        self.Camera3Thread.stop()
                    self.processHandle.terminate()
                else:
                    self.camActive = 1
                    if self.cameras.currentIndex() == 0:
                        self.Camera1Thread.start()
                        self.Camera1Thread.ImageUpdate.connect(self.updImage)
                    if self.cameras.currentIndex() == 1:
                        self.Camera2Thread.start()
                        self.Camera2Thread.ImageUpdate.connect(self.updImage)
                    if self.cameras.currentIndex() == 2:
                        self.Camera3Thread.start()
                        self.Camera3Thread.ImageUpdate.connect(self.updImage)
            else:
                self.prevCam = self.cameras.currentIndex()
                self.camActive = 1
                if self.cameras.currentIndex() == 0:
                    self.Camera2Thread.stop()
                    self.Camera3Thread.stop()
                    self.Camera1Thread.start()
                    self.Camera1Thread.ImageUpdate.connect(self.updImage)
                if self.cameras.currentIndex() == 1:
                    self.Camera1Thread.stop()
                    self.Camera3Thread.stop()
                    self.Camera2Thread.start()
                    self.Camera2Thread.ImageUpdate.connect(self.updImage)
                if self.cameras.currentIndex() == 2:
                    self.Camera1Thread.stop()
                    self.Camera2Thread.stop()
                    self.Camera3Thread.start()
                    self.Camera3Thread.ImageUpdate.connect(self.updImage)

    def startClicked(self):
        if self.camActive:
            if not self.processStarted:
                self.textBrowser.clear()
                ui.textBrowser.append(
                    datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
                self.textBrowser.append("НАЧИНАЮ ПРОЦЕСС")
                self.processHandle.start()
                self.processHandle.signal.connect(self.progUpd)
                self.processStarted = True
            else:
                ui.textBrowser.append(
                    datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
                self.textBrowser.append("ПРОЦЕСС ОСТАНОВЛЕН")
                self.lastProgressState = self.processProgress.value()
                self.processStarted = False
        else:
            self.textBrowser.append("НЕВОЗМОЖНО НАЧАТЬ ПРОЦЕСС\nКАМЕРА НЕ ВКЛЮЧЕНА")

    def stopClicked(self):
        if self.processStarted:
            self.processProgress.setValue(0)
            self.processHandle.terminate()
            ui.textBrowser.append(
                datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
            self.textBrowser.append("ПРОЦЕСС ОСТАНОВЛЕН")
            self.processStarted = False
    
    def labelUpd1(self, text, stylesheet):
        txt = text
        sth = stylesheet
        self.flame.setText(txt)
        self.flame.setStyleSheet(sth)
    
    def labelUpd2(self, text, stylesheet):
        txt = text
        sth = stylesheet
        self.motion.setText(txt)
        self.motion.setStyleSheet(sth)
        
    def labelUpd3(self, text, stylesheet):
        txt = text
        sth = stylesheet
        self.smoke.setText(txt)
        self.smoke.setStyleSheet(sth)
        
    def cpuUsage(self, value):
        val = value
        self.cpuLoad.setValue(val)

    def progUpd(self, value):
        val = value
        self.processProgress.setValue(val)

    def textBrowsUpd(self, text):
        txt = text
        self.textBrowser.append(txt)
                
    def displayTime(self):
        time = QTime.currentTime()
        text = time.toString('hh:mm:ss')
        self.lcdNumber.display(text)

    def updImage(self, Image):
        self.vidCapture.setPixmap(QPixmap.fromImage(Image))

    def loadCameras(self):
        cams = returnCameraIndexes()
        for i in cams:
            self.cameras.addItem("Камера " + str(i + 1))


    def retranslateUi(self, MainScreen):
        _translate = QtCore.QCoreApplication.translate
        MainScreen.setWindowTitle(_translate("MainScreen", "Process Monitor"))
        self.label.setText(_translate("MainScreen", "<html><head/><body><p align=\"center\">ВЫБЕРИТЕ КАМЕРУ</p></body></html>"))
        self.startVideo.setText(_translate("MainScreen", "ПУСК/СТОП ВИДЕО"))
        self.label_8.setText(_translate("MainScreen", "<html><head/><body><p><span style=\" font-size:12pt; font-weight:600;\">СОСТОЯНИЕ ДАТЧИКОВ</span></p></body></html>"))
        self.label_2.setText(_translate("MainScreen", "<html><head/><body><p><span style=\" font-size:11pt;\">ЗАДЫМЛЕНИЕ</span></p></body></html>"))
        self.smoke.setText(_translate("MainScreen", "<html><head/><body><p><span style=\" font-size:11pt;\">НОРМА</span></p></body></html>"))
        self.label_4.setText(_translate("MainScreen", "<html><head/><body><p><span style=\" font-size:11pt;\">ВОЗГОРАНИЕ</span></p></body></html>"))
        self.flame.setText(_translate("MainScreen", "<html><head/><body><p><span style=\" font-size:11pt;\">НОРМА</span></p></body></html>"))
        self.label_6.setText(_translate("MainScreen", "<html><head/><body><p><span style=\" font-size:11pt;\">ПРИСУТСТВИЕ</span></p></body></html>"))
        self.motion.setText(_translate("MainScreen", "<html><head/><body><p><span style=\" font-size:11pt;\">НОРМА</span></p></body></html>"))
        self.label_12.setText(_translate("MainScreen", "<html><head/><body><p><span style=\" font-size:11pt;\">ЗАГРУЗКА ЦП</span></p></body></html>"))
        self.cpuLoad.setFormat(_translate("MainScreen", "%p%"))
        self.label_10.setText(_translate("MainScreen", "СОСТОЯНИЕ ПРОЦЕССА"))

import backgrounds
import images

class Camera1Thread(QThread):
    ImageUpdate = pyqtSignal(QImage)
    def run(self):
        self.ThreadActive = True
        cap = cv2.VideoCapture(0)
        while self.ThreadActive:
            ret, img = cap.read()
            if ret:
                blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
                net.setInput(blob)
                layerNames = net.getLayerNames()
                outputNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
                outputs = net.forward(outputNames)
                if findObjects(outputs, img):
                    ui.isSomebodyHere = 1
                else:
                    ui.isSomebodyHere = 0
                rgbImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ConvertToQtFormat = QImage(rgbImage, 640, 480, QImage.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(640,480,Qt.KeepAspectRatio)
                self.ImageUpdate.emit(Pic)
                if self.ThreadActive == False:
                    cap.release()
            else:
                break
                self.stop()
    def stop(self):
        self.ThreadActive = False
        self.quit()

class Camera2Thread(QThread):
    ImageUpdate = pyqtSignal(QImage)
    def run(self):
        self.ThreadActive = True
        cap = cv2.VideoCapture(1)
        while self.ThreadActive:
            ret, img = cap.read()
            if ret:
                blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
                net.setInput(blob)
                layerNames = net.getLayerNames()
                outputNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
                outputs = net.forward(outputNames)
                if findObjects(outputs, img):
                    ui.isSomebodyHere = 1
                else:
                    ui.isSomebodyHere = 0
                rgbImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ConvertToQtFormat = QImage(rgbImage, 640, 480, QImage.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.ImageUpdate.emit(Pic)
                if self.ThreadActive == False:
                    cap.release()
            else:
                break
                self.stop()
    def stop(self):
        self.ThreadActive = False
        self.quit()

class Camera3Thread(QThread):
    ImageUpdate = pyqtSignal(QImage)
    def run(self):
        self.ThreadActive = True
        cap = cv2.VideoCapture(2)
        while self.ThreadActive:
            ret, img = cap.read()
            if ret:
                blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
                net.setInput(blob)
                layerNames = net.getLayerNames()
                outputNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
                outputs = net.forward(outputNames)
                if findObjects(outputs, img):
                    ui.isSomebodyHere = 1
                else:
                    ui.isSomebodyHere = 0
                rgbImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ConvertToQtFormat = QImage(rgbImage, 640, 480, QImage.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.ImageUpdate.emit(Pic)
                if self.ThreadActive == False:
                    cap.release()
            else:
                self.stop()
                break

    def stop(self):
        self.ThreadActive = False
        self.quit()


class CpuLoad(QThread):
    signal = pyqtSignal(int)
    flameSig = pyqtSignal(str, str)
    motionSig = pyqtSignal(str, str)
    smokeSig = pyqtSignal(str, str)
    def run(self):
        val = 0
        while True:
            val = psutil.cpu_percent(interval=1)
            #self.signal.emit(val)
            '''
            if (ui.MotionSensor == 1 or ui.isSomebodyHere == 1) and GPIO.input(23):
                ui.Fire = 1
                print("эвакуация")
                self.flameSig.emit("ОГОНЬ", "color: rgb(255, 0, 0);")
                ui.MotionSensor = 1
                self.motionSig.emit("ДВИЖЕНИЕ", "color: rgb(255, 0, 0);")
                pygame.mixer.music.load('FIRE_PEOPLE.mp3')
                pygame.mixer.music.play()
                time.sleep(20)
            elif GPIO.input(23):
                ui.Fire = 1
                self.flameSig.emit("ОГОНЬ", "color: rgb(255, 0, 0);")
                pygame.mixer.music.load('FIRE.mp3')
                pygame.mixer.music.play()
                time.sleep(20)
            else:
                self.flameSig.emit("НОРМА", "color: rgb(0, 255, 0);")
                pygame.mixer.music.stop()
                ui.MotionSensor = 0
                self.motionSig.emit("НОРМА", "color: rgb(0, 255, 0);")
                ui.Fire = 0
            if GPIO.input(26):
                ui.MotionSensor = 1
                self.motionSig.emit("ДВИЖЕНИЕ", "color: rgb(255, 0, 0);")
            else:
                ui.MotionSensor = 0
                self.motionSig.emit("НОРМА", "color: rgb(0, 255, 0);")
            if GPIO.input(27) == GPIO.LOW:
                ui.Smoke = 1
                self.smokeSig.emit("ДЫМ", "color: rgb(255, 0, 0);")
                pygame.mixer.music.load('SMOKE.mp3')
                pygame.mixer.music.play()
                time.sleep(15)
            else:
                ui.Smoke = 0
                self.smokeSig.emit("НОРМА", "color: rgb(0, 255, 0);")
                pygame.mixer.music.stop() '''
    def stop(self):
        #GPIO.cleanup()
        self.terminate()


class ProcessRun(QThread):
    signal = pyqtSignal(int)
    textbrowsersignal = pyqtSignal(str)
    def run(self):
        if ui.lastProgressState > 0:
            progVal = ui.lastProgressState
        else:
            progVal = 0
        if ui.camActive:
            if ui.isSomebodyHere == 0 and ui.Fire == 0 and ui.Smoke == 0 and ui.MotionSensor == 0:
                while True:
                    if not ui.processStarted:
                        break
                    while ui.processProgress.value() <= 100:
                        if ui.processStarted:
                            if ui.isSomebodyHere or ui.MotionSensor:
                                self.textbrowsersignal.emit(
                                    datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
                                self.textbrowsersignal.emit("РАБОЧИЙ В ОПАСНОЙ ЗОНЕ")
                                time.sleep(15)
                            elif ui.camActive == 0:
                                ui.processState = 0
                                ui.processStarted = False
                                self.textbrowsersignal.emit(
                                    datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
                                self.textbrowsersignal.emit("ПРОПАЛ СИГНАЛ С КАМЕРЫ...\nПРОЦЕСС ЗАВЕРШЁН")
                                ui.processState = 0
                                ui.processStarted = False
                            elif ui.Fire:
                                ui.processState = 0
                                ui.processStarted = False
                                self.signal.emit(0)
                                self.textbrowsersignal.emit(
                                    datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
                                self.textbrowsersignal.emit("ПОЖАР\nПРОЦЕСС АВАРИЙНО ЗАВЕРШЁН")
                            elif ui.Smoke:
                                ui.processState = 0
                                ui.processStarted = False
                                self.signal.emit(0)
                                self.textbrowsersignal.emit(
                                    datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
                                self.textbrowsersignal.emit("ЗАДЫМЛЕНИЕ\nПРОЦЕСС АВАРИЙНО ЗАВЕРШЁН")
                            else:
                                progVal += 1
                                time.sleep(0.25)
                                self.signal.emit(progVal)
                                if ui.processProgress.value() == 20:
                                    self.textbrowsersignal.emit(
                                        datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
                                    self.textbrowsersignal.emit("ПРОЦЕСС ЗАВЕРШЕН НА 20%")
                                if ui.processProgress.value() == 40:
                                    self.textbrowsersignal.emit(
                                        datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
                                    self.textbrowsersignal.emit("ПРОЦЕСС ЗАВЕРШЕН НА 40%")
                                if ui.processProgress.value() == 60:
                                    self.textbrowsersignal.emit(
                                        datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
                                    self.textbrowsersignal.emit("ПРОЦЕСС ЗАВЕРШЕН НА 60%")
                                if ui.processProgress.value() == 80:
                                    self.textbrowsersignal.emit(
                                        datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
                                    self.textbrowsersignal.emit("ПРОЦЕСС ЗАВЕРШЕН НА 80%")
                                if ui.processProgress.value() == 100:
                                    self.textbrowsersignal.emit(
                                        datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
                                    self.textbrowsersignal.emit("ПРОЦЕСС ЗАВЕРШЕН. \nВОЗВРАТ В ИСХОДНОЕ ПОЛОЖЕНИЕ")
                                    time.sleep(1.5)
                                    ui.processStarted = False
                                    self.signal.emit(0)
                                    self.stop()
                        else:
                            break
            elif ui.MotionSensor or ui.isSomebodyHere:
                self.textbrowsersignal.emit(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
                self.textbrowsersignal.emit("НЕВОЗМОЖНО НАЧАТЬ ПРОЦЕСС. \nРАБОЧИЙ В ЗОНЕ ОПАСНОСТИ")
            elif ui.Fire:
                self.textbrowsersignal.emit(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
                self.textbrowsersignal.emit("ОБНАРУЖЕНО ВОЗГОРАНИЕ\nНЕВОЗМОЖНО НАЧАТЬ ПРОЦЕСС.")
            elif ui.Smoke:
                self.textbrowsersignal.emit(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
                self.textbrowsersignal.emit("ОБНАРУЖЕНО ЗАДЫМЛЕНИЕ\nНЕВОЗМОЖНО НАЧАТЬ ПРОЦЕСС.")
        else:
            self.textbrowsersignal.emit(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
            self.textbrowsersignal.emit("КАМЕРА ДОЛЖНА БЫТЬ ВКЛЮЧЕНА")

    def stop(self):
        self.signal.emit(0)
        ui.processState = 0
        self.quit()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainScreen = QtWidgets.QMainWindow()
    ui = Ui_MainScreen()
    ui.setupUi(MainScreen)
    MainScreen.setFixedSize(MainScreen.size())
    timer = QTimer()
    timer.timeout.connect(ui.displayTime)
    timer.start(1000)
    ui.displayTime()
    MainScreen.show()
    sys.exit(app.exec_())

