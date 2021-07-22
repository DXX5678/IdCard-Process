from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSlot, QTextCodec
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QFrame, QGraphicsScene, QGraphicsPixmapItem, QMessageBox
from MainUI import Ui_MainWindow
import mainbynumpy as ClassifyAPI
import ImageClassifyViaText as ClassifyAPI2
import MyImage as CutImage
import MyText as ReadText
import cv2 as cv
from time import strftime, localtime
import numpy as np
import sys
import os

import qt5_applications

dirname = os.path.dirname(qt5_applications.__file__)
plugin_path = os.path.join(dirname, 'Qt', 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path

class CardImage:
    def __init__(self):
        self.ClassifyController = ClassifyAPI.ImageClassifyController()
        self.ImagePath = ["", ""]
        self.ImageType = ""
        self.ImageTypeReliability: float = 0.0  # 可信度
        self.cardInfo: list = []
        self.isFront: bool = False

    def getImageType(self, processedMatrix):
        self.cardInfo = []
        tempPath = os.getcwd() + "/CardR" + "/getImageTemp.jpg"
        cv.imencode('.jpg', processedMatrix)[1].tofile(tempPath)
        imageTypes, imageTypeReliabilities = self.ClassifyController.classify([tempPath])
        self.ImageTypeReliability = imageTypeReliabilities[0]
        print(imageTypes[0] + " " + str(imageTypeReliabilities))
        self.ImageType = imageTypes[0]
        # self.ImageTypeReliability = float(imageTypeReliabilities[0])
        if self.ImageType in {'身份证反面', '社保卡反面'}:
            self.isBack = True
        self.cardInfo.append(self.ImageType)
        self.cardInfo.append("当前分类可信度: " + str(self.ImageTypeReliability)[:4])


class UserInterface(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super(UserInterface, self).__init__(parent)
        self.setupUi(self)
        self.retranslateUi(self)

        self.currentImage = CardImage()
        self.currentImage.ImagePath[0] = os.getcwd() + "/CardR" + "/welcome.jpg"
        raw_data = np.fromfile(self.currentImage.ImagePath[0], dtype=np.uint8)
        # 先用numpy把图片文件存入内存：raw_data，把图片数据看做是纯字节数据
        self.cardImageMatrix = cv.imdecode(raw_data, cv.IMREAD_COLOR)  # 从内存数据读入图片
        self.ImagePresentation()

    @pyqtSlot()
    def on_select_clicked(self):
        # print("clicked")
        fileOpenWindow = QtWidgets.QMainWindow()
        imagePathTemp = QtWidgets.QFileDialog.getOpenFileName(fileOpenWindow, "选择文件", os.getcwd(),
                                                              "All Files(*)")
        print(imagePathTemp[0] + " " + imagePathTemp[1])  # debug
        if not imagePathTemp[0] == "":
            self.currentImage.ImagePath = imagePathTemp
            raw_data = np.fromfile(self.currentImage.ImagePath[0], dtype=np.uint8)
            # 先用numpy把图片文件存入内存：raw_data，把图片数据看做是纯字节数据
            self.cardImageMatrix = cv.imdecode(raw_data, cv.IMREAD_COLOR)  # 从内存数据读入图片
            self.ImagePresentation()  # 把导入的图片显示出来

    def ImagePresentation(self):
        x = self.cardImageMatrix.shape[1]  # 获取图像大小
        y = self.cardImageMatrix.shape[0]
        box_x = 1098
        box_y = 721
        img = cv.cvtColor(self.cardImageMatrix, cv.COLOR_BGR2RGB)  # 转换图像通道
        try:
            srcImgRatio = x / y
            boxRatio = box_x / box_y
            if srcImgRatio > boxRatio:
                img = cv.resize(img, (box_x, int(box_x / srcImgRatio)))
            else:
                img = cv.resize(img, (int(box_y * srcImgRatio), box_y))
        except Exception as reszException:
            print(str(reszException))
        zoomscale = 1  # 图片放缩尺度
        frame = QImage(img, img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        item = QGraphicsPixmapItem(pix)  # 创建像素图元
        item.setScale(zoomscale)
        scene = QGraphicsScene()  # 创建场景
        scene.addItem(item)
        self.srcimg.setScene(scene)  # 将场景添加至视图

    @pyqtSlot()
    def on_parse_clicked(self):
        try:
            if self.currentImage.ImagePath[0] == os.getcwd() + "/CardR" + "/welcome.jpg":
                QMessageBox.information(self, "提示", "请选择一张图片!", QMessageBox.Ok)
                return
            else:
                self.currentTextBrowser.clear()
            parseImageName = os.getcwd() + "/CardR" + "/TempForParse.jpg"
            cv.imencode('.jpg', self.cardImageMatrix)[1].tofile(parseImageName)
            processedImage = CutImage.MyImage.process(parseImageName)  # 切出来的卡片
            cut_flag = False
            if processedImage is not None:
                self.cardImageMatrix = processedImage
                self.currentImage.getImageType(processedImage)
                if self.currentImage.cardInfo[0] in {'身份证反面', '身份证正面', "社保卡反面", '社保卡正面'}:
                    self.cardImageMatrix = processedImage
                    self.ImagePresentation()
                if self.currentImage.cardInfo[0] in {'身份证反面', '身份证正面'}:
                    self.setAsIdCard()
                elif self.currentImage.cardInfo[0] == "社保卡反面":
                    self.setAsSeCardFront()
                else:
                    self.setAsAnyCard()
            else:
                self.currentImage.getImageType(self.cardImageMatrix)
                if not self.preClassifyViaText():
                    self.classifyViaText()
                    if self.currentImage.cardInfo[0] in {'身份证反面', '身份证正面'}:
                        self.setAsIdCard()
                    elif self.currentImage.cardInfo[0] == "社保卡反面":
                        self.setAsSeCardFront()
                    else:
                        self.setAsAnyCard()
                    cut_flag = True
                else:
                    if self.currentImage.cardInfo[0] != "手写签名":
                        self.currentImage.cardInfo[0] = "其它"
            self.cardInfoPresentation()  # 把导入的图片解析出的信息显示出来
            if cut_flag:
                QMessageBox.warning(self, "警告", "证件难以正常提取, 请重新选择!", QMessageBox.Ok)
        except Exception as parseException:
            cv.imshow("Processed Image Presentation", self.cardImageMatrix)
            self.currentImage.cardInfo.append("解析失败!")
            self.currentImage.cardInfo.append(str(parseException) + "\n")
            self.cardInfoPresentation()

    def cardInfoPresentation(self):
        self.textBrowser.append("------------" + strftime("%Y-%m-%d %H:%M:%S", localtime()) + "------------\n")
        self.textBrowser.append("当前图像: " + self.currentImage.ImagePath[0] + "\n")
        for singleInfo in self.currentImage.cardInfo:
            self.textBrowser.append(singleInfo)
            self.currentTextBrowser.append(singleInfo)
        self.textBrowser.append("\n-------------------------------------------------\n")
        self.currentImage.cardInfo = []

    @pyqtSlot()
    def on_exportcard_clicked(self):

        fileSaveWindow = QtWidgets.QMainWindow()
        dstImagePath = QtWidgets.QFileDialog.getSaveFileName(fileSaveWindow, "保存文件", os.getcwd(), "JPEG Files(*.jpg)")
        if dstImagePath[0] != "":
            cv.imencode('.jpg', self.cardImageMatrix)[1].tofile(dstImagePath[0])
            QMessageBox.information(self, "提示", "成功导出了当前图片!", QMessageBox.Ok)

    @pyqtSlot()
    def on_exportinfo_clicked(self):
        fileSaveWindow = QtWidgets.QMainWindow()
        dstTextPath = QtWidgets.QFileDialog.getSaveFileName(fileSaveWindow, "保存文件", os.getcwd(), "文本文件(*.txt)")
        if dstTextPath[0] != "":
            dstText = self.textBrowser.toPlainText()
            dstFile = open(dstTextPath[0], mode='w')
            dstFile.write(dstText)
            QMessageBox.information(self, "提示", "成功导出历史文本!", QMessageBox.Ok)
            dstFile.close()

    @pyqtSlot()
    def on_clearall_clicked(self):
        if QMessageBox.question(self, "提示", "清除右侧两个框内所有内容?", QMessageBox.Yes | QMessageBox.No,
                                QMessageBox.No) == QMessageBox.Yes:
            self.textBrowser.clear()
            self.currentTextBrowser.clear()

    @pyqtSlot()
    def on_rotate_clicked(self):
        self.cardImageMatrix = self.ImageRotate(self.cardImageMatrix)
        self.ImagePresentation()

    def setAsIdCard(self):
        if len(self.currentImage.cardInfo) > 3:
            return
        if self.currentImage.cardInfo[0] == '身份证反面':
            cardInfoList = ReadText.MyText.ocrprocess(self.cardImageMatrix)
            numsOfWords = len(cardInfoList)
            if numsOfWords == 4:
                for i in range(numsOfWords - 2, numsOfWords):
                    self.currentImage.cardInfo.append(cardInfoList[i])
            elif numsOfWords == 6:
                for i in range(numsOfWords - 3, numsOfWords, 2):
                    self.currentImage.cardInfo.append(cardInfoList[i])
            else:
                for i in range(numsOfWords):
                    self.currentImage.cardInfo.append(cardInfoList[i])
        elif self.currentImage.cardInfo[0] == '身份证正面':
            cardInfoList = ReadText.MyText.ocrprocess(self.cardImageMatrix)
            numsOfWords = len(cardInfoList)
            for i in range(numsOfWords):
                self.currentImage.cardInfo.append(cardInfoList[i])

    def setAsAnyCard(self):
        cardInfoList = ReadText.MyText.ocrprocess(self.cardImageMatrix)
        numsOfWords = len(cardInfoList)
        for i in range(numsOfWords):
            self.currentImage.cardInfo.append(cardInfoList[i])

    def setAsSeCardFront(self):
        return

    def preClassifyViaText(self) -> bool:
        try:
            cardInfoList = ReadText.MyText.ocrprocess(self.cardImageMatrix)
        except Exception as ocrException:
            self.currentImage.cardInfo.append("OCR failed!")
            self.currentImage.cardInfo.append(str(ocrException))
            return True
        modifiedClassifyCode: int = ClassifyAPI2.ImageClassify.getImageType(cardInfoList)
        return modifiedClassifyCode == -1

    def classifyViaText(self):
        try:
            cardInfoList = ReadText.MyText.ocrprocess(self.cardImageMatrix)
        except Exception as ocrException:
            self.currentImage.cardInfo.append("OCR failed!")
            self.currentImage.cardInfo.append(str(ocrException))
            return
        modifiedClassifyCode: int = ClassifyAPI2.ImageClassify.getImageType(cardInfoList)
        if modifiedClassifyCode != -1:
            self.currentImage.cardInfo = []  # 之前的写入作废
            if modifiedClassifyCode == 1:
                self.currentImage.cardInfo.append("身份证反面")
            elif modifiedClassifyCode == 2:
                self.currentImage.cardInfo.append("身份证正面")
            elif modifiedClassifyCode == 3:
                self.currentImage.cardInfo.append("社保卡反面")
            elif modifiedClassifyCode == 4:
                self.currentImage.cardInfo.append("社保卡正面")
            self.currentImage.cardInfo.append("当前分类可信度: 1.00(经文本识别)")
            self.currentImage.ImageTypeReliability = 1
            numsOfWords = len(cardInfoList)
            for i in range(numsOfWords):
                self.currentImage.cardInfo.append(cardInfoList[i])

    def ImageRotate(self, img_src):
        img_dst = np.rot90(img_src)
        return img_dst
