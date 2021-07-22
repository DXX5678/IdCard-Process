
import cv2 as cv

from ImageProcess import ImageProcess

"""对图片进行canny算子边缘检测"""


class EdgeHandle(ImageProcess):
    def __init__(self):
        pass

    def process_image(self, image, threshold1=8, threshold2=12, aperturesize=3, l2gradient=True):
        """
        :param image: 图片对象
        :param threshold1: 代表双阈值中的低阈值
        :param threshold2: 代表双阈值中的高阈值
        :param aperturesize: spbel核的窗口大小，默认为3*3
        :param l2gradient: 代表计算边缘梯度大小时使用的方式，True代表使用平方和开方的方式，False代表采用绝对值和的方式
        :return: 边缘检测后的图片
        """
        image = cv.Canny(image, threshold1, threshold2, apertureSize=aperturesize, L2gradient=l2gradient)
        return image

    """
    sobel算子边缘检测
    def process_image(self, image):
        image = cv.Sobel(image, ddepth=cv.CV_32F, dx=1, dy=1, ksize=3)
        image = np.abs(image)
        image = image / np.max(image)
        image = image * 255
        image = image.astype(np.uint8)
        return image
    """
