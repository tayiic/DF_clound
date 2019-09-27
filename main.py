#colab 备份用

#加载云盘
from google.colab import drive
drive.mount('/content/drive')
#改变工作目录
import os
os.chdir('/content/drive/My Drive/Colab Notebooks/DF/clound/')
!ls

#过12个小时 空间就会被自动清理
n = len(os.listdir('trainImg/'))
if n > 10000: 
    print("train有图像{}张".format(n))
else:
    os.chdir('/content/drive/My Drive/Colab Notebooks/DF/clound/trainImg')
    !ls
    !unzip -o Train.zip
    os.chdir('/content/drive/My Drive/Colab Notebooks/DF/clound/')

import tensorflow as tf
import numpy as np
import keras
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

Train_Img_Dir = 'trainImg/'
Train_Label_Path = 'Train_label.csv'
Test_Img_Dir = 'testImg/'
TFRecords_File = 'train.tfrecords'
submit_File = 'submit_example.csv'

CLASSES_num = 30 # 分类网络的类别数目，也是网络最后一层的单元数目
X_shape = [500, 500, 3] # 预处理后图像的大小
EPOCH = 100 # 训练多少轮
Batch_size = 128  # 训练的batch size

# tf.enable_eager_execution() #调试用




def read_image(name, type):
    """读图片 并把Image类型转成numpy
    name: 图片名
    type: ‘train' 'test'
    """
    if type == 'train':
        path = os.path.join(Train_Img_Dir, name)
    elif type == 'test': 
        path = os.path.join(Test_Img_Dir, name)

    img = Image.open(path) # img.mode = 'RGB'
    if img.mode != 'RGB':
        img = img.convert("RGB") #读取图片的过程中如果遇到非'RGB'就转换格式
    img_array = np.asarray(img, np.float32)
    # plt.imshow(img)  
    # plt.show()
    return img_array 

class DatasetGenerator:
    def __init__(self):
        self.train_labels = self._get_train_label() #获取图片名列表
        self.train_num = len(self.train_labels) #图片总数

    def write2TFRecoard(self, tfr_file=TFRecords_File, start=0, end=None, option=None):
        """
        #把图片生成TFRecords文件
        tfr_file: TFRecord文件的存放路径
        start, end: 保存图片的
        option: TFRecord文件保存的压缩格式
        """
        #start end参数检查 但是超出index范围没关系 
        trf_writer = tf.python_io.TFRecordWriter(tfr_file, options=option)

        for index, (name, label) in dg.train_labels[start:end].iterrows():
            img = read_image(name, 'train')
            img_shape = img.shape
            img = img.reshape(-1) #变成一维
            img = self.__preprocessing(img) #裁剪 归一化等预处理
            img_list = img.tolist() 
            feature_internal = {
                    'image_raw' : tf.train.Feature(float_list = tf.train.FloatList(value=img_list)), #内层feature编码方式
                    'img_shape' : tf.train.Feature(int64_list = tf.train.Int64List(value=img_shape)),
                    'label' : tf.train.Feature(int64_list = tf.train.Int64List(value=[int(label)]))     
                    }
            #使用tf.train.Example将features编码数据封装成特定的PB协议格式
            example = tf.train.Example(features=tf.train.Features(feature=feature_internal))
            #将序列化为字符串的example数据写入协议缓冲区
            trf_writer.write(example.SerializeToString())
        #关闭TFRecords文件操作接口    
        trf_writer.close()

    def get_from_TFRecoard(self, tfr_files=TFRecords_File):
        """
        ##从tfr_files指定的TFRecords文件，初始化一个dataset
        :param tfr_files: TFRecords文件路径
        :return: 
        """
        # 定义TFRecordDataset
        dataset = tf.data.TFRecordDataset(tfr_files) #默认一个文件
        #执行解析函数 得到数据集    
        dataset = dataset.map(self.__parse_function)
        # # 定义batch size大小，非常重要。
        # dataset = dataset.batch(Batch_size)
        # # 无限重复数据集
        # dataset = dataset.repeat()
        return dataset

    def __preprocessing(self, img):
        """对图像数据进行预处理"""
        # img_norm = tf.image.per_image_standardization(timg) #tensorflow中对图像标准化预处理的API

        img_norm = img/255.  #[0,1]归一化
        return img_norm

    def __parse_function(self, example_proto):
        """解析函数 
        :param example_proto: example序列化后的样本tf_serialized
        """
        features = {
            'image_raw' : tf.VarLenFeature(dtype=tf.float32),
            'img_shape' : tf.FixedLenFeature(shape=(3,), dtype=tf.int64),
            'label' : tf.FixedLenFeature(shape=(), dtype=tf.int64)
            }
        # 把序列化样本和解析字典送入函数里得到解析的样本
        parsed_example = tf.parse_single_example(example_proto, features) #返回字典
        # 解码 
        # 稀疏表示 转为 密集表示
        parsed_example['image_raw'] = tf.sparse_tensor_to_dense(parsed_example['image_raw'], default_value=0)
        # 转换tensor形状
        img_shape = parsed_example['img_shape']
        parsed_example['image_raw'] = tf.reshape(parsed_example['image_raw'], img_shape)
        
        # 如果使用dataset作为keras中，model.fit函数等的参数，则需要使用one_hot编码
        # 在tensorflow中，基本是不需要的，可以直接返回example['label']。
        one_hot_label = tf.one_hot(parsed_example['label'], CLASSES_num)
        
        return parsed_example['image_raw'], one_hot_label

    def _get_train_label(self):
        """获取所有的train label"""
        labels = pd.read_csv(Train_Label_Path)
        return labels

    def _get_train_img(self, start=0, end=100):
        """测试用 读图片"""
        for i in range(start, end):
            (name, label) = dg.train_labels.iloc[i]        
            img = read_image(name, 'train')
            yield(img, label)


