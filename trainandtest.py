from ImageClassify import ImageClassify
from imagehandle import ImageHandle


if __name__ == "__main__":
    # 要识别的图片处理后的长度
    length = 96
    # 要识别的图片处理后的宽度
    width = 96
    # 通道数
    channels = 3
    # 分类数
    classes = 6
    # 迭代次数
    epochs = 15
    # 模型保存路径
    model_save_path = "./imageclassify_save"
    # 训练集路径
    trainRootPath = "/home/fizz/Music/大学学习的知识总结/CardClassifier/CardClassifier/CNNmodel/classifyimages/train"
    # 测试集路径
    # testRootPath = "/home/fizz/Music/大学学习的知识总结/CardClassifier/CardClassifier/CNNmodel/classifyimages/test"

    image_classify = ImageClassify(length, width, channels, classes, model_save_path)
    image_handle = ImageHandle(length, width, channels)
    train_x, train_y = image_handle.get_trainxy(trainRootPath)
    # test_x, test_y = image_handle.get_trainxy(testRootPath)
    
    # 训练
    print("=============================训练=================================")
    image_classify.fit(train_x, train_y, epochs)

    # 测试
    print("=============================测试====================================")
    # print(image_classify.testbyarray(test_x, test_y))
