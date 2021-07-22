import cv2 as cv
from MyError import MyError

from ImageProcess import ImageProcess

"""图片内证件目标检测"""


class ConHandle(ImageProcess):
    def __init__(self):
        pass

    def process_image(self, image, image_r):
        """
        :param image: 图片对象
        :param image_r: 初始图片对象
        :return: 提取出的证件图片
        """
        h, w = image_r.shape[:2]
        image = cv.adaptiveThreshold(image, 255, 0, cv.THRESH_BINARY_INV, 3, 1)
        contours, hierarchy = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        num = len(contours)
        if num == 0:
            raise MyError("未检测到证件")
        results = []
        areaM = 0
        for i in range(num):
            areaM += cv.contourArea(contours[i], oriented=False)
        for i in range(num):
            area = cv.contourArea(contours[i], oriented=False)
            rect = cv.boundingRect(contours[i])
            #cv.rectangle(image_r, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0))
            #cv.imwrite("contour"+str(i)+".jpg", image_r)
            if area >= 100000:
                if (rect[2] / rect[3] <= 1.9) and (rect[2] / rect[3] >= 1.1):
                    expend1 = 40
                    expend2 = 20
                    results.append(image_r[max(rect[1] - expend2, 0):min(rect[1] + rect[3] + expend2, h),
                                   max(rect[0] - expend1, 0):min(rect[0] + rect[2] + expend1, w)])
                    break
            else:
                if area >= (0.3 * areaM):
                    if (rect[2] / rect[3] <= 1.9) and (rect[2] / rect[3] >= 1.1):
                        expend1 = 40
                        expend2 = 20
                        results.append(image_r[max(rect[1] - expend2, 0):min(rect[1] + rect[3] + expend2, h),
                                   max(rect[0] - expend1, 0):min(rect[0] + rect[2] + expend1, w)])
                        break
        if len(results) == 0:
            raise MyError("未检测到证件")
        image_r = results[0]
        image = results[0]
        return image, image_r
