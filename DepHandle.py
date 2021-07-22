import cv2 as cv

from ImageProcess import ImageProcess

"""保持原图长宽比进行图片压缩"""


class DepHandle(ImageProcess):
    def __init__(self):
        pass

    def process_image(self, image, image_r, size=800):
        """
        :param image: 图片对象
        :param image_r: 初始图片对象
        :param size: 压缩后的图片宽度
        :return: 压缩后的图片
        """
        image = cv.resize(image, (size, int(image.shape[0] / image.shape[1] * size)))
        image_r = cv.resize(image_r, (size, int(image.shape[0] / image.shape[1] * size)))
        return image, image_r
