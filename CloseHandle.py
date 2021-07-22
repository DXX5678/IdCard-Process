
import cv2 as cv

from ImageProcess import ImageProcess

"""图片闭运算"""


class CloseHandle(ImageProcess):
    def __init__(self):
        pass

    def getKernel(self, shape=cv.MORPH_RECT, size=(1, 2)):
        """
        :param shape: 卷积核的形状
        :param size: 卷积核的大小
        :return: 卷积核
        """
        kernel = cv.getStructuringElement(shape, size)
        return kernel

    def process_image(self, image, count1=5, count2=4):
        """
        :param image: 图片对象
        :param count1: 腐蚀操作次数
        :param count2: 膨胀操作次数
        :return: 处理后图片
        """
        kernel = self.getKernel()
        image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel, iterations=count1)
        """可选择在闭运算后二次膨胀"""
        kernel = self.getKernel(size=(2, 2))
        image = cv.dilate(image, kernel, iterations=count2)
        return image
