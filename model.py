from tensorflow import keras


class ImageClassifyModelFactory:
    @staticmethod
    def getModel(length, width, channels, classes):
        model = keras.Sequential(
            # [
            # keras.layers.Flatten(input_shape=(length, width, channels)),
            # keras.layers.Dense(128, activation='relu'),
            # keras.layers.Dense(classes)]
        )

        # 卷积
        model.add(keras.layers.Conv2D(32, (channels, channels), activation = 'relu', input_shape = (length, width, channels)))
        # 池化
        model.add(keras.layers.MaxPooling2D(2,2))
        model.add(keras.layers.Conv2D(64, (channels, channels), activation = 'relu'))
        model.add(keras.layers.MaxPooling2D(2, 2))
        model.add(keras.layers.Conv2D(64, (channels, channels), activation = 'relu'))

        # 转成1维
        model.add(keras.layers.Flatten())

        # 传给全连接层
        model.add(keras.layers.Dense(64, activation = 'relu'))
        model.add(keras.layers.Dense(classes, activation = 'sigmoid'))

        model.summary()

        return model

"""
输入层 width * height * channels
卷积层 特征提取
池化层 降维
线性整流层 使用活化函数
全连接层 输出, 计算得分

"""