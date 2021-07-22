class ImageClassify:
    def __init__(self, textList):
        self.textList: list = textList

    def getType(self):
        for eachWord in self.textList:
            if "民身份" in eachWord:
                if len(eachWord) < 10:
                    return 1    # 识别为国徽面
                else:
                    return 2    # 识别为信息面
            elif "会保障" in eachWord:
                if len(self.textList) < 3:
                    return 3    # 识别为社保卡反面
                else:
                    return 4    # 识别为社保卡个人信息面
        return -1

    @staticmethod
    def getImageType(textList):
        currentImageClassify = ImageClassify(textList)
        return currentImageClassify.getType()
