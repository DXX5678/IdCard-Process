
import cv2 as cv

from ImageProcess import ImageProcess

"""图片灰度化"""


class GrayHandle(ImageProcess):
    def __init__(self):
        pass

    def process_image(self, image):
        """
        :param image: 图片对象
        :return: 灰度化后图片
        """
        image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        rows = image.shape[0]
        cols = image.shape[1]
        for i in range(rows):
            for j in range(cols):
                image_gray[i, j] = image[i, j][0] * 0.114 + image[i, j][1] * 0.578 + image[i, j][2] * 0.299
        return image_gray
