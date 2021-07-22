import copy

import cv2 as cv
import numpy as np

from MyError import MyError
from CloseHandle import CloseHandle
from BinaryHandle import BinaryHandle
from ConHandle import ConHandle
from EdgeHandle import EdgeHandle
from GrayHandle import GrayHandle
from GausHandle import GausHandle
from DepHandle import DepHandle
from RecHandle import RecHandle


class MyImage:
    def __init__(self, image):
        self.image = image
        self.image_r = copy.deepcopy(image)

    def prefilter(self, imageProcess):
        self.image = imageProcess.process_image(self.image)
        return self

    def filter(self, imageProcess):
        self.image, self.image_r = imageProcess.process_image(self.image, self.image_r)
        return self

    @staticmethod
    def process(image_path):
        raw_data = np.fromfile(image_path, dtype=np.uint8)
        # 先用numpy把图片文件存入内存：raw_data，把图片数据看做是纯字节数据
        image = cv.imdecode(raw_data, cv.IMREAD_COLOR)  # 从内存数据读入图片
        try:
            myimage = MyImage(image)
            myimage.image, myimage.image_r = DepHandle().process_image(myimage.image, myimage.image_r)
            myimage.prefilter(GausHandle()).prefilter(EdgeHandle()).prefilter(BinaryHandle()).prefilter(CloseHandle())
            myimage.filter(ConHandle()).filter(RecHandle())
            return myimage.image_r
        except MyError as e:
            return None


if __name__ == '__main__':
    for i in range(1):
        # image = cv.imread(str(i + 1) + ".jpg")
        try:
            """
            myimage = MyImage(image)
            myimage.image, myimage.image_r = DepHandle().process_image(myimage.image, myimage.image_r)
            myimage.prefilter(GausHandle()).prefilter(EdgeHandle()).prefilter(BinaryHandle()).prefilter(CloseHandle())
            myimage.filter(ConHandle()).filter(RecHandle())
            cv.imwrite(str(i + 25) + ".jpg", myimage.image_r)
            """
            image_r = MyImage.process(str(i + 1) + ".jpg")
            cv.imwrite(str(i + 26) + ".jpg", image_r)
        except MyError as e:
            print(e.value)
            exit(1)
