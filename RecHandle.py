import copy
import math
import numpy as np
import cv2 as cv
from EdgeHandle import EdgeHandle
from GausHandle import GausHandle
from ImageProcess import ImageProcess

"""对提取的证件进行倾斜校正"""


class RecHandle(ImageProcess):
    def __init__(self):
        pass

    def process_image(self, image, image_r):
        """
        :param image: 图片对象
        :param image_r: 初始图片对象
        :return: 校正后的图片
        """
        result = []
        h, w = image_r.shape[:2]
        image = GausHandle().process_image(image, (31, 31))
        # image = GrayHandle().process_image(image)
        image = EdgeHandle().process_image(image, 8, 11)
        lines = cv.HoughLines(image, rho=1, theta=math.pi / 180, threshold=90)
        if lines is None:
            return image, image_r
        uppoint = [float("inf"), 0]
        downpoint = [0, 0]
        leftpoint = [float("inf"), 0]
        rightpoint = [0, 0]
        num = 0
        for line in lines:
            num += 1
            rho = line[0][0]
            theta = line[0][1]
            """
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = rho * a
            y0 = rho * b  # 原点到直线的垂线，与直线的交点
            x1 = int(x0 + 1000 * (-b))  # 取1000长度，在（x0, y0）上下从直线中各取一点 (由于图像坐标系y轴反向，所以为-b)
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)
            cv.line(image_r, (x1, y1), (x2, y2), (0, 0, 255), 2)
            """
            if (theta <= (math.pi * 1. / 5.)) or (theta >= (4. * math.pi / 5.0)):
                if abs(rho * math.cos(theta)) < abs(leftpoint[0]):
                    leftpoint[0] = rho
                    leftpoint[1] = theta
                if abs(rho * math.cos(theta)) > abs(rightpoint[0]):
                    rightpoint[0] = rho
                    rightpoint[1] = theta
            elif (theta >= (math.pi * 4. / 10.)) and (theta <= (6. * math.pi / 10.0)):
                if abs(rho * math.sin(theta)) < abs(uppoint[0]):
                    uppoint[0] = rho
                    uppoint[1] = theta
                if abs(rho * math.sin(theta)) > abs(downpoint[0]):
                    downpoint[0] = rho
                    downpoint[1] = theta
        if num < 4:
            return image, image_r
        """
        cv.imshow("rec.jpg", image_r)
        key = cv.waitKey(0)
        if key == 27:
            cv.destroyAllWindows()
        """
        rho1 = uppoint[0]
        theta1 = uppoint[1]
        rho2 = leftpoint[0]
        theta2 = leftpoint[1]
        """
        a = math.cos(theta1)
        b = math.sin(theta1)
        x0 = rho1 * a
        y0 = rho1 * b  # 原点到直线的垂线，与直线的交点
        x1 = int(x0 + 1000 * (-b))  # 取1000长度，在（x0, y0）上下从直线中各取一点 (由于图像坐标系y轴反向，所以为-b)
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        cv.line(image_r, (x1, y1), (x2, y2), (0, 0, 255), 2)

        a = math.cos(theta2)
        b = math.sin(theta2)
        x0 = rho2 * a
        y0 = rho2 * b  # 原点到直线的垂线，与直线的交点
        x1 = int(x0 + 1000 * (-b))  # 取1000长度，在（x0, y0）上下从直线中各取一点 (由于图像坐标系y轴反向，所以为-b)
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        cv.line(image_r, (x1, y1), (x2, y2), (0, 0, 255), 2)
        """
        A = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
        b = np.array([[rho1], [rho2]])
        x0, y0 = np.linalg.solve(A, b)
        if math.isnan(x0) or math.isnan(y0):
            return image, image_r
            # raise MyError("图片矫正出错")
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        result.append([x0, y0])
        rho2 = rightpoint[0]
        theta2 = rightpoint[1]
        """
        a = math.cos(theta2)
        b = math.sin(theta2)
        x0 = rho2 * a
        y0 = rho2 * b  # 原点到直线的垂线，与直线的交点
        x1 = int(x0 + 1000 * (-b))  # 取1000长度，在（x0, y0）上下从直线中各取一点 (由于图像坐标系y轴反向，所以为-b)
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        cv.line(image_r, (x1, y1), (x2, y2), (0, 0, 255), 2)
        """
        A = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
        b = np.array([[rho1], [rho2]])
        x0, y0 = np.linalg.solve(A, b)
        if math.isnan(x0) or math.isnan(y0):
            return image, image_r
            # raise MyError("图片矫正出错")
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        result.append([x0, y0])
        rho1 = downpoint[0]
        theta1 = downpoint[1]
        rho2 = leftpoint[0]
        theta2 = leftpoint[1]
        """
        a = math.cos(theta1)
        b = math.sin(theta1)
        x0 = rho1 * a
        y0 = rho1 * b  # 原点到直线的垂线，与直线的交点
        x1 = int(x0 + 1000 * (-b))  # 取1000长度，在（x0, y0）上下从直线中各取一点 (由于图像坐标系y轴反向，所以为-b)
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        cv.line(image_r, (x1, y1), (x2, y2), (0, 0, 255), 2)

        a = math.cos(theta2)
        b = math.sin(theta2)
        x0 = rho2 * a
        y0 = rho2 * b  # 原点到直线的垂线，与直线的交点
        x1 = int(x0 + 1000 * (-b))  # 取1000长度，在（x0, y0）上下从直线中各取一点 (由于图像坐标系y轴反向，所以为-b)
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        cv.line(image_r, (x1, y1), (x2, y2), (0, 0, 255), 2)
        """
        A = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
        b = np.array([[rho1], [rho2]])
        x0, y0 = np.linalg.solve(A, b)
        if math.isnan(x0) or math.isnan(y0):
            return image, image_r
            # raise MyError("图片矫正出错")
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        result.append([x0, y0])
        rho2 = rightpoint[0]
        theta2 = rightpoint[1]
        """
        a = math.cos(theta2)
        b = math.sin(theta2)
        x0 = rho2 * a
        y0 = rho2 * b  # 原点到直线的垂线，与直线的交点
        x1 = int(x0 + 1000 * (-b))  # 取1000长度，在（x0, y0）上下从直线中各取一点 (由于图像坐标系y轴反向，所以为-b)
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        cv.line(image_r, (x1, y1), (x2, y2), (0, 0, 255), 2)
        """
        A = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
        b = np.array([[rho1], [rho2]])
        x0, y0 = np.linalg.solve(A, b)
        """
        cv.imshow("rec.jpg", image_r)
        key = cv.waitKey(0)
        if key == 27:
            cv.destroyAllWindows()
        """
        if math.isnan(x0) or math.isnan(y0):
            return image, image_r
            # raise MyError("图片矫正出错")
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        result.append([x0, y0])
        """
        if result[0][0] < 0:
            result[0][0] = 0
        if result[0][1] < 0:
            result[0][1] = 0
        if result[1][0] > w:
            result[1][0] = w
        if result[1][1] < 0:
            result[1][1] = 0       
        """

        length = math.sqrt(math.pow(result[1][0] - result[0][0], 2) + math.pow(result[1][1] - result[0][1], 2)) / 1.55

        if result[2][0] < 0 or result[2][1] > h:

            tz = leftpoint[1]
            if tz > (math.pi / 2.):
                result[2][0] = result[0][0] + length * math.sin(tz)
                result[2][1] = result[0][1] + length * (-math.cos(tz))
            else:
                result[2][0] = result[0][0] - length * math.sin(tz)
                result[2][1] = result[0][1] + length * math.cos(tz)

        if result[3][0] > w or result[3][1] > h:

            ty = rightpoint[1]
            if ty > (math.pi / 2.):
                result[3][0] = result[1][0] + length * math.sin(ty)
                result[3][1] = result[1][1] + length * (-math.cos(ty))
            else:
                result[3][0] = result[1][0] - length * math.sin(ty)
                result[3][1] = result[1][1] + length * math.cos(ty)
        lengthw = math.sqrt(math.pow(result[3][0] - result[2][0], 2) + math.pow(result[3][1] - result[2][1], 2)) / 1.55
        if abs(length - lengthw) > 25:
            # print(1)
            if length < lengthw:
                tz = leftpoint[1]
                if tz > (math.pi / 2.):
                    result[2][0] = result[0][0] + length * math.sin(tz)
                    result[2][1] = result[0][1] + length * (-math.cos(tz))
                else:
                    result[2][0] = result[0][0] - length * math.sin(tz)
                    result[2][1] = result[0][1] + length * math.cos(tz)
                ty = rightpoint[1]
                if ty > (math.pi / 2.):
                    result[3][0] = result[1][0] + length * math.sin(ty)
                    result[3][1] = result[1][1] + length * (-math.cos(ty))
                else:
                    result[3][0] = result[1][0] - length * math.sin(ty)
                    result[3][1] = result[1][1] + length * math.cos(ty)
            else:
                tz = leftpoint[1]
                if tz > (math.pi / 2.):
                    result[0][0] = result[2][0] - lengthw * math.sin(tz)
                    result[0][1] = result[2][1] - lengthw * (-math.cos(tz))
                else:
                    result[0][0] = result[2][0] + lengthw * math.sin(tz)
                    result[0][1] = result[2][1] - lengthw * math.cos(tz)
                ty = rightpoint[1]
                if ty > (math.pi / 2.):
                    result[1][0] = result[3][0] - lengthw * math.sin(ty)
                    result[1][1] = result[3][1] - lengthw * (-math.cos(ty))
                else:
                    result[1][0] = result[3][0] + lengthw * math.sin(ty)
                    result[1][1] = result[3][1] - lengthw * math.cos(ty)
        lengthup = math.sqrt(math.pow(result[1][0] - result[0][0], 2) + math.pow(result[1][1] - result[0][1], 2))
        lengthle = math.sqrt(math.pow(result[2][0] - result[0][0], 2) + math.pow(result[2][1] - result[0][1], 2))
        lengthri = math.sqrt(math.pow(result[3][0] - result[1][0], 2) + math.pow(result[3][1] - result[1][1], 2))
        # print(lengthup, lengthle)
        if (lengthup / lengthle) < 1.4 or (lengthup / lengthri) < 1.4:
            # print(2)
            tz = leftpoint[1]
            if tz > (math.pi / 2.):
                result[2][0] = result[0][0] + length * math.sin(tz)
                result[2][1] = result[0][1] + length * (-math.cos(tz))
            else:
                result[2][0] = result[0][0] - length * math.sin(tz)
                result[2][1] = result[0][1] + length * math.cos(tz)
            ty = rightpoint[1]
            if ty > (math.pi / 2.):
                result[3][0] = result[1][0] + length * math.sin(ty)
                result[3][1] = result[1][1] + length * (-math.cos(ty))
            else:
                result[3][0] = result[1][0] - length * math.sin(ty)
                result[3][1] = result[1][1] + length * math.cos(ty)

        if abs(result[1][0] - result[0][0]) < w * 1 / 3:
            return image, image_r
        if abs(result[2][1] - result[0][1]) < h * 1 / 3:
            return image, image_r
        pts1 = np.float32([result[0], result[1], result[2], result[3]])
        pts2 = np.float32([[0, 0], [713, 0], [0, 450], [713, 450]])
        M = cv.getPerspectiveTransform(pts1, pts2)
        image_r = cv.warpPerspective(image_r, M, (713, 450))
        # print(result)
        return image, image_r
