import os
from ocr import ocr
import cv2 as cv
import re


class MyText:
    def __init__(self, image):
        self.image = image

    def single_pic_proc(self):
        """
        :return: 返回文本字典，索引为其下标值{0:[array(), text], 1:[array(), text], 2:[]}
        """
        result, image_framed = ocr(self.image)
        return result

    @staticmethod
    def ocrprocess(image):
        mytext = MyText(image)
        result = mytext.single_pic_proc()
        texts = []
        if len(result) >= 8:
            idx_start = 0
            idx_end = 0
            for key in result:
                result[key][1] = re.sub(
                    "\\【.*?】+|\\《.*?》+|\\#.*?#+|[﹒!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]",
                    "", result[key][1])
                result[key][1] = result[key][1].replace(" ", "")
                if "姓名" in result[key][1]:
                    idx_start = key
                    continue
                if result[key][1].isdigit() or len(result[key][1]) == 19:
                    idx_end = key
                    break
            for key in result:
                if idx_start <= key <= idx_end:
                    if len(result[key][1]) < 4:
                        continue
                    texts.append(result[key][1])
            if len(texts) == 5:
                texts.pop(3)
            if len(texts) == 6:
                texts.pop(2)
                texts.pop(3)

        else:
            for key in result:
                result[key][1] = re.sub(
                    "\\【.*?】+|\\《.*?》+|\\#.*?#+|[﹒!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]",
                    "", result[key][1])
                result[key][1] = result[key][1].replace(" ", "")
                texts.append(result[key][1])

        return texts


if __name__ == '__main__':
    """需要图片路径"""
    for i in range(1):
        mytext = MyText(cv.imread(r"C:\Users\xuxiao\Desktop\1.jpg"))

        result = mytext.ocrprocess(mytext.image)
        print(result)
        print("............................")
        """
        result = mytext.single_pic_proc()
        for key in result:
            result[key][1] = re.sub(
                "\\【.*?】+|\\《.*?》+|\\#.*?#+|[﹒!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]",
                "", result[key][1])
            result[key][1] = result[key][1].replace(" ", "")
            print(result[key][1])
        print("............................")
        """
