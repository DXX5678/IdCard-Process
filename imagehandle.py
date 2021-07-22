# 图片处理框架
import pathlib
from random import random
import random
import tensorflow as tf
import numpy


class ImageHandle:
    def __init__(self, image_length, image_width, channels=1):
        """
        图片处理器
        :param image_length: 图片长度
        :param image_width: 图片宽度
        :param channels: 图片通道数目 黑白照片设定为1
        """
        self.image_length = image_length
        self.image_width = image_width
        self.channels = channels
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE

    def load_and_preprocess_image(self, path):
        """
        回调函数, 加载并处理图片, 包括尺寸缩放, 归1化
        :param path: 图片路径
        :return: void
        """
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, self.channels)
        image = tf.image.resize(image, [self.image_length, self.image_width])
        image /= 255.0
        return image

    def getLength(self, path):
        """
        获得路进下图片长度
        :param path:
        :return:
        """
        data_root = pathlib.Path(path)
        all_image_paths = list(data_root.glob("*/*"))
        all_image_paths = [str(path) for path in all_image_paths]
        return len(all_image_paths)

    def get_predict_image_numpy(self, path_arr):
        all_images_paths = path_arr
        images_arrays = list(map(self.load_and_preprocess_image, all_images_paths))
        images_arrays_numpy = numpy.asarray(images_arrays, numpy.float32)
        return images_arrays_numpy

    def index_to_label(self, path):
        """
        获得文件名和分类索引的对应关系{'0', 'file_name'}
        :param path: 数据集名称
        :return:
        """
        return {0: '身份证正面', 1: '身份证反面', 2: '社保卡正面', 3: "社保卡反面", 4: "手写签名"}
        # 文件根路径
        #data_root = pathlib.Path(path)
        #all_image_paths = list(data_root.glob("*/*"))
        #all_image_paths = [str(path) for path in all_image_paths]

        # 标签处理
        #label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
        #index_to_label = dict((index, name) for index, name in enumerate(label_names))
        #return index_to_label

    def get_trainxy(self, path):
        """
        生成数组形式的数据集合
        :param path:
        """
        # 文件根路径
        data_root = pathlib.Path(path)
        all_image_paths = list(data_root.glob("*/*"))
        all_image_paths = [str(path) for path in all_image_paths]

        # 图片处理
        images_arrays = list(map(self.load_and_preprocess_image, all_image_paths))

        # 标签处理
        label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
        label_to_index = dict((name, index) for index, name in enumerate(label_names))
        all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]

        # 打乱
        images_arrays_numpy = numpy.asarray(images_arrays, numpy.float32)
        all_image_labels_numpy = numpy.asarray(all_image_labels, numpy.int64)
        count = images_arrays_numpy.shape[0]
        ran = numpy.arange(count)
        numpy.random.shuffle(ran)
        images_arrays_numpy = images_arrays_numpy[ran]
        all_image_labels_numpy = all_image_labels_numpy[ran]

        return images_arrays_numpy, all_image_labels_numpy

    def get_dataset(self, path, batch_size = 32):
        """
        获得dataset
        :param path: 文件的路径
        :param batch_size: 每次拿多少图片
        :return: tf.dataset
        """
        # 文件根路径
        data_root = pathlib.Path(path)
        all_image_paths = list(data_root.glob("*/*"))
        all_image_paths = [str(path) for path in all_image_paths]
        random.shuffle(all_image_paths)
        path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
        # 加载图片并处理
        image_ds = path_ds.map(self.load_and_preprocess_image, num_parallel_calls = self.AUTOTUNE)
        print(type(image_ds))
        print(image_ds)
        # 标签处理
        label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
        label_to_index = dict((name, index) for index, name in enumerate(label_names))
        all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]
        label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))

        # 图片和标签处理成元组的集合
        image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

        BATCH_SIZE = batch_size
        # 设置一个和数据集大小一致的 shuffle buffer size（随机缓冲区大小）以保证数据
        # 被充分打乱。
        image_count = len(all_image_paths)
        ds = image_label_ds.shuffle(buffer_size=image_count)
        ds = ds.repeat()
        ds = ds.batch(BATCH_SIZE)
        # 当模型在训练的时候，`prefetch` 使数据集在后台取得 batch。
        ds = ds.prefetch(buffer_size=self.AUTOTUNE)
        ds = image_label_ds.apply(
           tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
        ds = ds.batch(BATCH_SIZE)
        return ds

