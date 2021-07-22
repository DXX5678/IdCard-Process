
import cv2 as cv

from ImageProcess import ImageProcess

"""图片二值化处理"""


class BinaryHandle(ImageProcess):
    def __init__(self):
        pass

    def process_image(self, image, max_value=255, method=cv.ADAPTIVE_THRESH_GAUSSIAN_C, value=cv.THRESH_BINARY_INV, size=3,
                      num_c=1):
        """
        :param image: 图片对象
        :param max_value: 最大值
        :param method: 自适应方法
        :param value: 二值化方法
        :param size: 窗口大小
        :param num_c: 常数值
        :return: 处理后图片
        """
        image = cv.adaptiveThreshold(image, max_value, method, value, size, num_c)
        return image
