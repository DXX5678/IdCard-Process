from ImageClassify import ImageClassify
from imagehandle import ImageHandle
#from classifyimages.ImageClassify import ImageClassify
#from classifyimages.imagehandle import ImageHandle

import numpy
import os

class ImageClassifyController:
    def __init__(self):
        # 要识别的图片处理后的长度
        self.length = 128
        # 要识别的图片处理后的宽度
        self.width = 128
        # 通道数
        self.channels = 3
        # 分类数
        self.classes = 5
        # 迭代次数
        self.epochs = 13
        # 模型保存路径
        # self.model_save_path = "./imageclassify_save"
        self.model_save_path = os.path.join(os.path.dirname(__file__), 'imageclassify_save')
        # 训练集路径
        self.trainRootPath = r"C:\Users\一凡\Desktop\deep-learning\图像识别\入门项目\classifyimages\train"
        # 测试集路径
        self.testRootPath = "/home/fizz/Music/大学学习的知识总结/CardClassifier/CardClassifier/CNNmodel/classifyimages/test"
        self.image_classify = ImageClassify(self.length, self.width, self.channels, self.classes, self.model_save_path)
        self.image_handle = ImageHandle(self.length, self.width, self.channels)
        self.index_to_label = self.image_handle.index_to_label(self.trainRootPath)
        try:
            if not self.image_classify.is_trained:
                train_x, train_y = self.image_handle.get_trainxy(self.trainRootPath)
                # 训练
                self.image_classify.fit(train_x, train_y, self.epochs)
        except Exception as initException:
            return

    def classify(self, image_paths):
        """
        给出图片绝对路径数组, 进行分类
        :param image_paths: 图片绝对路径的数组
        """
        image_numpy = self.image_handle.get_predict_image_numpy(image_paths)
        result = self.image_classify.predictbyarray(image_numpy)
        labels = []
        rate = []
        for np in result:
            index = numpy.argmax(np)
            r = numpy.max(np)
            label = self.index_to_label[index]
            avg = np.var()
            # print(np)
            # print(avg)
            label = self.index_to_label[index]
            rate.append(r)
            labels.append(label)

        return labels, rate

if __name__ == '__main__':
    """
    images_paths = [
        "/home/fizz/Music/大学学习的知识总结/CardClassifier/CardClassifier/CNNmodel/classifyimages/test/id_card_b/1.jpg",
        "/home/fizz/Music/大学学习的知识总结/CardClassifier/CardClassifier/CNNmodel/classifyimages/test/id_card_b/6.jpg",
        "/home/fizz/Music/大学学习的知识总结/CardClassifier/CardClassifier/CNNmodel/classifyimages/test/she_card_b/1.jpg",
        "/home/fizz/Music/大学学习的知识总结/CardClassifier/CardClassifier/CNNmodel/classifyimages/test/she_card_b/1.jpg",
        "/home/fizz/Music/大学学习的知识总结/CardClassifier/CardClassifier/CNNmodel/classifyimages/test/she_card_b/4.jpg",
        "/home/fizz/Music/大学学习的知识总结/CardClassifier/CardClassifier/CNNmodel/classifyimages/test/she_card_b/5.jpg",
        "/home/fizz/Music/大学学习的知识总结/CardClassifier/CardClassifier/CNNmodel/classifyimages/test/she_card_b/9.jpg",
        "/home/fizz/Downloads/5.jpg",
        "/home/fizz/Downloads/6.jpg",
        "/home/fizz/Downloads/7.jpg",
        "/home/fizz/Downloads/8.jpg"
    ]
    """
    images_paths = [
        r"C:\Users\xuxiao\Documents\Code\2021\Python\deep-learning\UserInterface\getImageTemp.jpg"
    ]
    controller = ImageClassifyController()
    print(controller.classify(images_paths))