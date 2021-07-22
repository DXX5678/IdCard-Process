from model import ImageClassifyModelFactory as modelFactory
# from classifyimages.model import ImageClassifyModelFactory as modelFactory

import tensorflow as tf
from PIL import Image


class ImageClassify:

    def __init__(self, length, width, channels, classes, model_save_path):
        """
        :param length: 图片长度
        :param width: 图片宽度
        :param channels: 通道数目 黑白照片为1, 彩色照片为3
        :param classes: 分类数目
        """
        # 尝试从文件加载模型
        try:
            restored_keras_model = tf.keras.models.load_model(model_save_path)
            self.model = restored_keras_model
            self.is_trained = True

        except Exception:
            self.model = modelFactory.getModel(length, width, channels, classes)
            self.is_trained = False
            self.__compile()

        self.length = length
        self.width = width
        self.channels = channels
        self.classes = classes
        self.model_save_path = model_save_path

    def __compile(self):
        """
        编译模型
        """
        self.model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

    def train(self, dataset, epochs, steps_per_epoch):
        """
        训练模型
        :param dataset:
        :param epochs: 迭代次数
        :param steps_per_epoch: 步骤, 默认为3
        """
        return self.model.fit(dataset, epochs = epochs, steps_per_epoch = steps_per_epoch)

    def trainAndTest(self, train_dataset, test_dataset, epochs, steps_per_epoch, validation_steps):
        """
        训练并测试
        :param train_dataset:
        :param test_dataset:
        :param epochs:
        :param steps_per_epoch:
        :param validation_steps:
        :return:
        """
        return self.model.fit(
            train_dataset, epochs=epochs,
            validation_data=test_dataset,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps)

    def predict(self, dataset, steps):
        return self.model.predict(dataset, steps = steps)

    def test(self, dataset, steps_per_epoch):
        """
        测试模型
        :param dataset:
        :return:
        """
        return self.model.evaluate(dataset, steps=steps_per_epoch)

    def fit(self, x, y, epochs):
        """
        训练模型
        :param x: 训练数组
        :param y: 测试数组
        :param epochs: 迭代次数
        :return: 损失率, 准确率
        """
        if self.is_trained:
            return

        self.model.fit(x, y, epochs=epochs)
        self.model.save(self.model_save_path)
        self.is_trained = True

    def testbyarray(self, x, y):
        """
        测试
        :param x: test_x
        :param y: test_y
        :return:
        """
        return self.model.evaluate(x, y, verbose = 2)

    def predictbyarray(self, x):
        """
        预测
        :param x: test_x
        :return:
        """
        return self.model.predict(x)