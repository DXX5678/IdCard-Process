
import cv2 as cv

from ImageProcess import ImageProcess

"""图片高斯模糊化"""


class GausHandle(ImageProcess):
    def __init__(self):
        pass

    def process_image(self, image, size=(23, 23), x_sta_dev=0, y_sta_dev=0, bordertype=0):
        """
        :param image: 图片对象
        :param size: 高斯卷积核大小
        :param x_sta_dev: 一维水平方向高斯卷积核的标准差
        :param y_sta_dev: 一维垂直方向高斯卷积核的标准差
        :param bordertype: 填充边界类型
        :return: 处理后的图片
        """
        image = cv.GaussianBlur(image, size, x_sta_dev, y_sta_dev, bordertype)
        return image
