from tensorflow import keras
from keras import backend as K
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, AveragePooling2D, Dropout, Concatenate, Add, ZeroPadding2D, UpSampling2D
from tensorflow.keras.layers import Flatten, BatchNormalization, Activation, Input, SeparableConv2D, LeakyReLU, DepthwiseConv2D
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.applications import vgg16, vgg19, resnet50
from tensorflow.keras import regularizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.train import Features, Feature, Example, Int64List, FloatList, BytesList
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import pandas as pd
import os
import sys
import time
from functools import wraps

def timeit(func):
    """@装饰器 计算func消耗的时间"""
    def wrapper(*args, **kwargs):
#         start = time.time()
        ret = func(*args, **kwargs)
#         cost = time.time() - start
#         print('【{}】  cost {:.2f}\'S'.format(func.__name__, cost))
        return ret
    return wrapper 


"""****************************************"""
""" 处理图片                 """
"""****************************************"""
TRAIN_IMG_DIR = 'trainImg/' #训练图片的目录
TEST_IMG_DIR = 'testImg/'  #测试图片的目录
TEST_LIST_FILE = 'lib/submit_example.csv' #test图片文件名和label的列表文件
X_SHAPE = [416, 416, 3] # 预处理后图像的大小
CLOUND = {    #编号（type）  云状类型
    0:'不是云', # 
    1:'中云-高积云-絮状高积云',
    2:'中云-高积云-透光高积云',
    3:'中云-高积云-荚状高积云',
    4:'中云-高积云-积云性高积云',
    5:'中云-高积云-蔽光高积云',
    6:'中云-高积云-堡状高积云',
    7:'中云-高层云-透光高层云',
    8:'中云-高层云-蔽光高层云',
    9:'高云-卷云-伪卷云',
    10:'高云-卷云-密卷云',
    11:'高云-卷云-毛卷云',
    12:'高云-卷云-钩卷云',
    13:'高云-卷积云-卷积云',
    14:'高云-卷层云-匀卷层云',
    15:'高云-卷层云-毛卷层云',
    16:'低云-雨层云-雨层云',
    17:'低云-雨层云-碎雨云',
    18:'低云-积云-碎积云',
    19:'低云-积云-浓积云',
    20:'低云-积云-淡积云',
    21:'低云-积雨云-鬃积雨云',
    22:'低云-积雨云-秃积雨云',
    23:'低云-层云-碎层云',
    24:'低云-层云-层云',
    25:'低云-层积云-透光层积云',
    26:'低云-层积云-荚状层积云',
    27:'低云-层积云-积云性层积云',
    28:'低云-层积云-蔽光层积云',
    29:'低云-层积云-堡状层积云',
    }

class Pic:
    @staticmethod
    def read_image(path):
        """
        @@@ 读图片  
        :params
            path: 文件路径
        :return 
            PIL.Image类型的图片
        """
        try:
            img = Image.open(path) # img.mode = 'RGB'
        except Exception as e:
            print('read_image: {} error...'.format(path))
            raise e
        if img.mode != 'RGB':
            img = img.convert("RGB") #读取图片的过程中如果遇到非'RGB'就转换格式
        return img     

    @staticmethod
    def show_double_img(before, after, num=3):
        """
        @@@ 并列显示处理前和处理后的图片 用于比较
        :param
            before: list 处理前PIL.Image 
            after: list 处理后PIL.Image
            num:一次显示几行 
        :return
            None
        """
        plt.figure(figsize=(14, 10), dpi=75, ) #画布 宽1400 高700
        for i in range(len(before)):
            p = i % num + 1 #在第几个小图显示
            plt.subplot(2, num, p)
            plt.imshow(before[i])        
            plt.subplot(2, num, p + num)
            plt.imshow(after[i])          
            if (i+1)%num == 0 or i==len(before)-1: 
                plt.show()            
        
    @staticmethod
    def preprocessing(img, region='big', aug=1):
        """
        @@@ 对图像数据进行预处理
        :param
            img: Image格式的图片
            region: 裁剪区域 ‘small'取原图最小范围
            aug: int 数据增强 几倍 ，i.e.2表示多生成1倍图片
        :return
            list: 里面是 norm后的array [H,W,3]，值范围[-0.5,-0.5]
        """
        img_list = []  #存放返回结果
        #其他处理。。。

        assert (aug>=1 and isinstance(aug, int)), "augmentation 输入参数不对"
        for i in range(aug):
            #数据增强时 裁剪区域稍微移动一下 让图片有点不一样
            shift = (-1)**i*0.02
            #四周裁剪掉的百分比(基于裁剪后的尺寸) ‘big'最接近原图
            if region == 'small':
                left, top = 0.08-2*shift, 0.1-2*shift
                right, bottom = 0.08+2*shift, 0.25+2*shift 
            elif region == 'middle':
                left, top = 0.05-shift, 0.03-shift 
                right, bottom = 0.05+shift, 0.12+shift 
            elif region == 'big':
                left, top = 0.03-shift, 0.02-shift 
                right, bottom = 0.03+shift, 0.06+shift 
            else:
                left, top, right, bottom = 0, 0, 0, 0 #默认不裁边
            
            #resize crop变成H*W大小
            image = Pic._resize(img, region=[left, top, right, bottom])
            if i%2 == 1:  #偶数次时 水平翻转图片
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
           
            #PIL Image转换为array
            img_array = np.asarray(image, np.float32)
            # img_norm = tf.image.per_image_standardization(timg) #tensorflow中对图像标准化预处理的API
            img_norm = img_array/255. - 0.5  #[-0.5,0.5]归一化
           
            img_list.append(img_norm)

        return img_list

    @staticmethod
    def _resize(img, region=[0, 0, 0, 0]):
        """
        @@@ 对图像裁切 缩减
        :param
            img: PIL.Image
            region: 四周裁剪掉的百分比区域  默认不裁边
        :return
            PIL.Image 预定义大小的图片 
        """
        left, top, right, bottom = region
        #最终输出尺寸
        H, W = X_SHAPE[:2]  
        #resize后的图片尺寸
        exH = int(H * (1 + top + bottom)) #高
        exW = int(W * (1 + left + right)) #宽
        h, w = img.size[::-1]  #原始图片的高 宽
        new_size = None #用来缩放的新尺寸（宽高）img.resize用宽在前面
        if h > w:  #如果原图宽小
            if w > exW:  #图片超大
                new_size = (exW, int(h*exW/w)) #宽缩到resize尺寸 高相应缩
            elif w < W:  #图片超小
                new_size = (W, int(h*W/w))  #宽放大到最终尺寸 高同步放大
            else: image = img  #图片不大不小 不变
        else: #原图高比较小
            if h > exH:  #图片超大
                new_size = (int(w*exH/h), exH)
            elif h < H: #图片超小
                new_size = (int(w*H/h), H)
            else: image = img
            
        if new_size: #改变图片大小
            image = img.resize(new_size, Image.ANTIALIAS) #压缩质量 改更好的？ 
        #对resize好的图片 进行裁剪，满足预定义尺寸H,W （根据天空的实际图片确定的策略）
        h, w = image.size[::-1] #resize后的尺寸
        top = int(top * h) #上高裁切 
        left = int(left * w) #左边裁切
        image = image.crop((left, top, left+W, top+H)) #四周裁剪 留下H*W区域
        return image
 
    @staticmethod
    def get_class_num():
        """
        @@@ 获取单标签的分类总数 也就是最后层神经元数目
        :return
            int
        """
#         NLs = Pic.get_train_NLs('single')
#         num = len(NLs['Code'].unique())
        return len(CLOUND)

class Pre_Model:
    def __init__(self):
        print('使用了预训练模型')
        self.vgg16_para = (18, (13, 13, 512))  #最后输出

    def vgg16(self):
        """
        @ 创建 模型
        :param
            None 
        :return 
            model
        """
        model_input = Input(shape=X_SHAPE)         
        #先加载预定义的模型
        base_model = vgg16.VGG16(include_top=False,
                        weights='imagenet',
                        input_tensor=model_input,
                        )
        for layer in base_model.layers[:]:
            layer.trainable = False  #darknet部分 权重不训练
#         base_model.summary()
#         for i, layer in enumerate(base_model.layers):
#             print(i, layer)
        #输出的层
        predictions = base_model.layers[self.vgg16_para[0]].output
        model = Model(model_input, predictions)
        print('加载了预训练模型 【 vgg16】')
        return model


"""****************************************"""
""" 处理数据                 """
"""****************************************"""
NUM_PER_TFRECOARD = 2000 #每个tfrecoard文件包括的图片数
CLASS_NUM = Pic.get_class_num()  #单标签 标签类别数目
if os.path.exists('/content/drive/My Drive/DF/clound/lib/'):
    TFR_DIR = '/content/drive/My Drive/DF/clound/lib/'  #google
else:
    TFR_DIR = 'lib/'  #tfrecoard文件的存放路径
#用colab时 每次都重新生成文件 可以考虑缩小样本 0.1=1/10 None=不缩小
COLAB_PIC_RATIO = 1 
BATCH_SIZE = 12  #批量读取 1bacth多少张照片
LABEL_CLASS = {  #train数据集的分类
            'xs' : '单label训练集',
            'vs' : '单label验证集',
            'xm' : '多label训练集',           
            'vm' : '多label验证集',    
    }
STEP_PER_EPOCH = {}  #一个epoch需要多少步 总图片/batch_size

class DatasetGenerator:
    def __init__(self, base_model=None, augmentation=1):
        """
        :Args 
            base_model: str 预训练模型，  None表示没有
            augmentation: int 数据增强 几倍 ，i.e.2表示多生成1倍图片
        """
        self.model_name = base_model
        if base_model:
            self.model = eval('Pre_Model().' + base_model + '()')  #预训练模型
            self.model_para = eval('Pre_Model().' + base_model + '_para')  #预训练模型参数
        else:
            self.model = None
        if augmentation >= 1 and isinstance(augmentation, int):        
            self.augment = augmentation  #数据增强倍数
        else:
            print('augmentation={} 输入错误'.format(augmentation))
            sys.exit()   
        self.batch_size = BATCH_SIZE
        #已打乱了的train数据集列表（包括单label 多label)
        self.train_label = self.__get_shuffled_train_lable()
        #单label训练集和验证集的文件列表
        self.xs_NLs, self.vs_NLs = self.__get_train_NLs('single')
        #多label训练集和验证集的文件列表
        self.xm_NLs, self.vm_NLs = self.__get_train_NLs('multi')
         
        #检查文件夹的tfrecoard文件，若没有则生成新的
#         self.check_tfrecoard()
        
        #计算step_per_epoch  fit时候用
        self.__calc_step_per_epoch()
        #标记（临时策略） 用于_deal_label 区分单label还是多label
        self.__now_working_class = None
        #标记 用于_parse_function 返回只包括img和label，还是包括全部信息
        self.__dataset_for_train = True #True 表示返回训练用的格式(img label)


    def check_tfrecoard(self):
        """
        @ 检查tfrecoard文件，若没有则生成新的
            ps: 有base_model，生成对应model的输出，如果没有，直接生成图片的array
        :param
        :return 
            None
        """       
        #检查每一个数据集的tfrecoards是否存在
        for lclass in list(LABEL_CLASS.keys())[:]:
            print('{}检查 TFRecoard:'.format(lclass))
            df = self.get_NLs(lclass)  #文件名df
            num = self.__calc_tfr_num(df, lclass)  #需要几个tfr
            size = len(df) // num + 1  #平均每个tfr最多保存几个图片
            
            All_existed = True  #tfr文件是否都存在
            for i in range(num):
                path = self.__create_tfr_name(i+1, lclass)
                if not os.path.isfile(path):
                    All_existed = False
                    break 
                else:
                    print(' [{}]文件存在  pass...'.format(path))
            
            if All_existed: continue
            
            print(' [{}]文件缺少，数据增强倍数{}，正全部重新生成{}个tfrecoard...'.format(
                LABEL_CLASS[lclass], self.augment, num))
            for i in range(num):
#                     print('"{}"：{}张图片 准备create TFRecoard:{}'.format(lclass, len(df), fnames))
                path = self.__create_tfr_name(i+1, lclass)
                df_slice = df.iloc[i*size:(i+1)*size]
                print('\n  写入 [{}] df[{}:{}] 数据X{}...'.format(
                    path, i*size, (i+1)*size, self.augment))
                if self.__create_tfrecoard(path, df_slice):
                    print('  Succeed!')
                else:
                    print(' Faile...sys.exit()')
                    sys.exit()           

        print('OK! 【{}】的所有TFRecoard文件检查完毕！\n'.format(LABEL_CLASS.keys()))

    @timeit    
    def __create_tfrecoard(self, tfr, df):
        """
        @@@ 把输入的图片列表 生成一个tfrecoard文件  若存在则覆盖
        :param
            tfr：生成的tfrecoard文件 名
            df：要保存的图片 FileName和lable的列表
        :return 
            True成功  False失败
        """            
#         print('---- 准备创建tfrecoard文件【{}】...'.format(tfr, len(df)))
        ok_num = 0  #成功写入tfrecoard的图片数目
        #考虑用 with tf.io.TFRecordWriter(tfr, options=option) as writer:
        with tf.io.TFRecordWriter(tfr) as tfr_writer:
            for index, (fname, flabel) in df.iterrows():
                try:
                    img = Pic.read_image(fname)
                except:
                    print('---- {}不存在，跳过...'.format(fname))
                    continue  #文件不存在 忽略 处理下一个
#                 print(index, img)
                flabel = [int(x) for x in flabel.split(';')]
                #对多label的情况，把label固定到5个长度（batch要统一长度）  即一张图片最多5种云
                if len(flabel) > 1:
                    flabel += [0,0,0]
                    flabel = flabel[:5]
                name = [tf.compat.as_bytes(fname)]
                 
                #裁剪 归一化等预处理  默认返回一个 有图像增强时，返回多个
                imgs = Pic.preprocessing(img, 'small', aug=self.augment)  #[] 
                x_shape = imgs[0].shape #处理后的图片尺寸 等于输入到模型中的图片尺寸
                assert x_shape == tuple(X_SHAPE), "准备写入TRF的图片尺寸不等于{}".format(X_SHAPE)
                for img in imgs:
                    if self.model:  #如果有预训练模型
                        img = np.expand_dims(img, 0)
                        output = self.model.predict(img)
#                         print(output[0, 0,0, :15])
                        image = output.reshape(-1)
#                         print('预训练后，', output.shape, image.shape)
                    else:
                        image = img.reshape(-1) #变成一维
                    #内层feature编码方式
                    # print('内层feature编码', name, flabel)
                    feature_internal = {
                                'image_raw' : Feature(float_list = FloatList(value=image)), #图形数据
                                'name'  : Feature(bytes_list = BytesList(value=name)), #路径文件名
                                'img_shape' : Feature(int64_list = Int64List(value=x_shape)),  
                                'label'   : Feature(int64_list = Int64List(value=flabel)),  #字符串的label   
                                }
                    #使用tf.train.Example将features编码数据封装成特定的PB协议格式
                    example = Example(features=Features(feature=feature_internal))
                    try:
                        #将序列化为字符串的example数据写入协议缓冲区
                        tfr_writer.write(example.SerializeToString())
                    except:
                        print('---- {}图片处理 保存过程中出错.'.format(fname))
                        return False
                    else:
                        ok_num += 1
                        if ok_num == len(df) * self.augment - 1: print('|||')
#                         if ok_num%30 == 0: print(' *', end="") #

        print('\n---- 创建了 [{}]，共写入图片{}张'.format(tfr, ok_num))
        return True    
    
    def get_NLs(self, lclass):
        """@@@获取文件名df 根据数据集类别"""
        return eval('self.' + lclass + '_NLs')
    
    def __calc_tfr_num(self, df, lclass):
        """@@@计算需要几个tfrecoard存储  根据数据集类别"""
        _len = len(df) * self.augment  #数据增强后
        return (_len - 1) // NUM_PER_TFRECOARD + 1

    def __create_tfr_name(self, i, lclass):
        """
        @@@ 生成tfrecoard文件名字  
        :Args
            i: 第几个
            lclass: 
                'xs' 'vs' 单label训练集和验证集 
                'xm' 'vm' 多label训练集和验证集   
        :return
            str (带路径)
        """
        extra = '_'  #文件命名用
        if self.model_name: 
            extra += self.model_name + '_'
        name = TFR_DIR + lclass + extra + str(i) + '.tfrecord'
        return name

    def __get_tfr_name(self, lclass):
        """@获取tfrecoard文件名字"""
        df = self.get_NLs(lclass)
        num = self.__calc_tfr_num(df, lclass)
        return [self.__create_tfr_name(i+1, lclass) for i in range(num)]
                        
    @timeit
    def __get_train_NLs(self, lclass):
        """
        @@@ 通过lclass 获取对应数据集的图片 FileName和lable的列表
        :param
            lclass: 
                single: 单label训练集和验证集 
                multi: 多label训练集和验证集 
        :return
            dataframe: 训练集和验证集的label按比例分割 随机打乱
                single: ['xs' 'vs']; 'multi':['xm' 'vm']
        """
        df = self.train_label
        
        if lclass == 'single':  #单label的(图片名和label列表)
            NLs = df[df.apply(lambda x: len(x['Code'])<=2, axis=1)].copy()
            path =  TRAIN_IMG_DIR 
        elif lclass == 'multi':  #多label  
            NLs = df[df.apply(lambda x: len(x['Code'])>2, axis=1)].copy()
            path = os.path.join(TRAIN_IMG_DIR, 'multi/')
        else:
            raise RuntimeError('输入错误 lclass= "{}"'.format(lclass) )
        #文件名上添加路径
        NLs['FileName'] = NLs['FileName'].apply(lambda x: os.path.join(path, x))   

        ratio = 0.7  #切割比例 训练集的大小 70%  验证集30%
        if lclass == 'single':
            #各个label class统计的数量
            desc = NLs.groupby(NLs['Code']).count().to_dict()['FileName'] 
            #按比例 确保训练集和验证集 各类标签都均匀取到
            index_list = {}  #df的index列表
            label = {}  #label class计数
            for index, row in NLs.iterrows():
                c = row['Code']
                n = label.get(c, 0)  #这个类别已经取出来了几个
                if n > desc[str(c)]*ratio: #这个类别已经取到了足够的数量 
                    index_list[index] = False
                else:
                    index_list[index] = True
                    label[c] = n+1
                    
            index = pd.Series(index_list)
            x = NLs[index].reset_index(drop=True)
            y = NLs[~index].reset_index(drop=True)
        else:  #多label的 就简单分割
            trunc = int(len(NLs) * ratio) #训练集大小
            x = NLs[:trunc].reset_index(drop=True)
            y = NLs[trunc:].reset_index(drop=True)         

        return x, y
 
    def __get_test_NLs(self, ratio=1):
        """
        @@@ 获取test所有的image的列表(name,label)
        :param
            ratio: 比例0-1 返回全部数据的百分之多少 
        :return 
            dataframe 
        """
        df = pd.read_csv(TEST_LIST_FILE)
        #文件名上添加路径
        df['FileName'] = df['FileName'].apply(lambda x: TEST_IMG_DIR + x)    
        #为了应付colab经常删文件，故缩小样本节约时间,正常不要这样操作
        return df[:int(ratio * len(df))]
    
    
    def __get_shuffled_train_lable(self):
        """
        @@@ 获取打乱了的train数据列表，若没有则生成新的
        :param
             None
        :return 
            dataframe
        """     
        train_label = 'lib/Train_label.csv' #训练图片文件名和label的列表文件     
        assert os.path.isfile(train_label) , "lib/Train_label.csv 不存在"    
        
        shuffled_train_label = 'lib/Shuffled_train_label.csv'
        if os.path.isfile(shuffled_train_label):
            df = pd.read_csv(shuffled_train_label)
            return df
        else:
            print('"{}"不存在，准备创建...'.format(shuffled_train_label))
            df = pd.read_csv(train_label)
            #打乱顺序 frac是要返回的比例 1=100%
            df = df.sample(frac=1).reset_index(drop=True) 
            #重复一次 可能会更乱
            df = df.sample(frac=1).reset_index(drop=True)
            #写入硬盘 让这个df和接下来生成的tfrecoard对应起来 
            df.to_csv(shuffled_train_label, index=False)  #生成csv文件
            return df
        
     
    @timeit    
    def read_data_from_TFRecoard(self, lclass, batch_size=BATCH_SIZE):
        """
        @@@ 获取数据集数据 从tfrecoard里读取
        :param
            lclass: train数据集的分类
        :return 
            dataset  
            [batch,H,W,3] if batch， 3D[H,W,3] if no batch
            sys.exit() if no file to read
        """
        assert lclass in LABEL_CLASS.keys(), "lcalss 输入error..."
        fnames = self.__get_tfr_name(lclass) #文件名列表
        self.__now_working_class = lclass  #传递给__deal_label()
        
        tfr_files = [] #待读取的tfrecoard文件列表
        for f in fnames:
            if os.path.exists(f): tfr_files.append(f)
        if tfr_files == []: 
            print('{}没有读取到任何tfrecoard...sys.exit()'.format(lclass))
            sys.exit()
#         print(f'{lclass}需要文件是{fnames}')
        print('【{}】找到并读取了{}个TFRecoard...'.format(lclass, len(tfr_files)))
        dataset_raw = tf.data.TFRecordDataset(tfr_files)
        # Set the number of datapoints you want to load and shuffle 
        buf_size = len(self.get_NLs(lclass)) * self.augment // len(fnames)
#         print('buf_size', buf_size)
        dataset = dataset_raw.shuffle(buffer_size = buf_size)
        #To decode the message use the tf.train.Example.FromString method.
        #example_proto = tf.train.Example.FromString(serialized_example)
        #执行解析函数 得到数据集    
        dataset = dataset.map(self.__parse_function)
        # 不加参数=无限重复数据集
        dataset = dataset.repeat()                   
        # 定义batchsize大小
        if batch_size: #batch时 必须保证各元素长度都一样
            dataset = dataset.batch(batch_size)
        return dataset

    def iterator_from_tfrecoard(self, lclass):
        """
        @ 获取iterator 数据集数据从tfrecoard里读取
        :param
            lclass: train数据集的分类
        :return 
            interator 可迭代的image label数据
            [batch,H,W,3] if batch， 3D[H,W,3] if no batch
            None if no file to read
        """
        dataset = self.read_data_from_TFRecoard(lclass, self.batch_size)
        iterator = dataset.make_one_shot_iterator()
        # return iterator
        return iterator.get_next()

    def __parse_function(self, example_proto):
        """
        @@@ 解析函数  解析dataset的每一个Example
        :param 
            example_proto: 单个example序列化后的样本
        :return
            img: array[H,W,C]
            name: b''
            label: [int]
        """
        feature_description  = {
            'image_raw' : tf.io.VarLenFeature(dtype=tf.float32),
            'name' : tf.io.VarLenFeature(dtype=tf.string),
            'img_shape' : tf.io.FixedLenFeature(shape=(3,), dtype=tf.int64),
            'label' : tf.io.VarLenFeature(dtype=tf.int64)
            }
        #把序列化样本和解析字典送入函数里得到解析的样本        
        parsed_example = tf.io.parse_single_example(example_proto, feature_description ) #返回字典
        #解码 
        img = tf.sparse.to_dense(parsed_example['image_raw'], default_value=0) # 稀疏->密集表示
        if self.model_name:  #有预训练
            image = tf.reshape(img, self.model_para[1]) #
        else:
            #img = tf.reshape(img, parsed_example['img_shape']) # 转换为原来形状
            #没搞定 用tf返回 DatasetV1Adapter shapes: (?, ?, ?) 先用固定尺寸来转换
            image = tf.reshape(img, X_SHAPE) # 转换为原来形状
        label = self.__deal_label(parsed_example['label'])
        name = tf.sparse.to_dense(parsed_example['name'], default_value='')
        name = tf.squeeze(name,)
        #如果有batch 则必须保证各item返回各元素长度都一样，才能加入一个batch内   
        if self.__dataset_for_train:
            return image, label 
        else:
            return image, label, name
            
    def __deal_label(self, label):
        """
        @@@ 把原始字符串的label 变成int
        :param
            label：原始label 单label 或者多label(e.g. 1:12)
        :return 
            list 单lable是one_hot  多label暂时未处理[int]  
        """
        lclass = self.__now_working_class  #现在正在处理哪个数据集
        assert lclass in LABEL_CLASS.keys(), "lcalss 输入error..."
        label = tf.sparse.to_dense(label, default_value=0)
        if lclass == 'xs' or lclass == 'vs': #如果是单label 直接转换为one-hot
            label = tf.one_hot(label[0], CLASS_NUM)
        else:  #如果是多label  还没想好
            print('现在是多label数据集"{}"，暂时不处理label'.format(lclass))
        return label       
 
    def __calc_step_per_epoch(self):
        """
        @@@ 计算每个label class的一个epoch需要几步能遍历
        :param
            None
        :return 
            None 保存到
        """    
        for c in LABEL_CLASS.keys():
            #简单一点 暂时用dataframe的列表，而不是真正的文件里面的图片数，实际上应该差不多
            df = self.get_NLs(c) 
            STEP_PER_EPOCH[c] = len(df) * self.augment // BATCH_SIZE + 1
            
    def __get_clound_name(self, labels):
        """
        @@@ 根据label 获取对应云的名字
        :param
            labels: int的list
        :return 
            list 云的名字
        """        
        names = []
        for label in labels:
            names.append(CLOUND[label])
        return names

    def _test_read_write(self):
        """@测试整个流程"""
        sess =  tf.compat.v1.InteractiveSession() 
        sess.run(tf.compat.v1.global_variables_initializer())
        
        self.__dataset_for_train = False #flag dataset读取全部类别
        classes = ['xs', 'vs', 'xm', 'vm'][:]
        for c in classes:
            print('测试{} "{}" 读取tfrecoard...'.format(c, LABEL_CLASS[c]))
            dataset = self.read_data_from_TFRecoard(c)
            iter = tf.compat.v1.data.make_one_shot_iterator(dataset)
            data = iter.get_next()
            if self.__dataset_for_train:
                image, label = data
            else:
                imgs, labels, names = data
#             self.__test_print(imgs, labels, names)
    
            #如果vgg16 用下面代码测试
            print('label, name', labels.eval()[0], names.eval()[0])
            imgs = imgs.eval()[0]
            print('img data;', imgs[0,0, :15])
            
    def _test_preprocessing(self):
        """@@@ 测试 preprocessing函数""" 
        df = self.xs_NLs[51:58]
        before, after = [], [] 
        for index, (fname, label) in df.iterrows():
            img = Pic.read_image(fname)
            imgs = Pic.preprocessing(img, 'big', 2)[0]
            before.append(imgs[0])
            after.append(imgs[1])
        Pic.show_double_img(before, after)

    def _test_resize(self):
        """@@@ 测试 resize函数""" 
        before, after = [], [] 
        df = self.xs_NLs[:20]
        for index, (fname, label) in df.iterrows():
            img = Pic.read_image(fname)
            before.append(img)
            ret = Pic._resize(img, region=[0.1, 0.08, 0.08, 0.25])
            after.append(ret)
        print(after)
        Pic.show_double_img(before, after)

    def __test_print(self, imgs, labels, names): 
        """@测试 显示iterator的元素"""         
        p_freq = {} #打印频率 
        try:
            for i in range(2):
                image = imgs.eval()+0.5
                name = names.eval()
                label = labels.eval()
                print('  batch {}: '.format(i+1))
                for j in range(image.shape[0])[:]:
                    p_freq[name[j]] = p_freq.get(name[j], 0) + 1
                    print('  No.{} image name:{}  label:{} '.format(j+1, name[j], label[j]  ))
                    # xname = name[j].decode('utf-8')  #图片上显示的xlable名字
                    # for n in self.__get_clound_name(label[j]):
                    #     xname += " " + n   
                    # plt.xlabel(xname, fontproperties="SimHei")
                    plt.imshow(image[j])
                    plt.show()
        except tf.errors.OutOfRangeError:
            print("end!读完了...")        

        print('{}张图片调用频率：\n'.format(len(p_freq)))
        for key, value in p_freq.items():
            print('{value} : {key}'.format(key = key, value = value))
        
# weights_kernel = os.path.join(TFR_DIR, 'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')

EPOCHS = 100  #fit训练次数

def showinfo(func):
    """@装饰器 显示辅助信息 shape param等"""
    def wrapper(*args, **kwargs):
        # print('调用', func.__name__, args[1], 'args:',args, 'kw:',kwargs)
        res = func(*args, **kwargs)
        return res
    return wrapper

class MyModel():
    def __init__(self):
        self.model = self.create_model_vgg16_base() 

    """
    model的组件
    """
    @showinfo
    @wraps(Conv2D)
    def __Conv2D(self, x, *args, **kwargs):
        """
        @ 2D卷积层
        :param
            x: input 
            args[0]: Conv2D filters
            args[1]: Conv2D kernels
            kwargs: 可选 strides ...
        :return 
            Output tensor after applying `Conv2D` 
        """ 
        conv_kwargs = {'kernel_regularizer': regularizers.l2(5e-4)} #加点正则化 
        #'strides'==(2,2) 相当于pool
        conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same' #最好还是传'padding'
        conv_kwargs.update(kwargs)
        return Conv2D(*args, **conv_kwargs)(x)

    @showinfo
    def __Conv2D_BN(self, x, *args, **kwargs):
        """
        @ 2D卷积层 + batch-normalization + LeakyReLU
        :param
            x: input 
            args[0]: Conv2D filters
            args[1]: Conv2D kernels
        :return 
            Output tensor after applying `Conv2D` and `BatchNormalization`
        """  
        no_bias_kwargs = {'use_bias': False}
        no_bias_kwargs.update(kwargs)
        x = self.__Conv2D(x, *args, **no_bias_kwargs) 
        x = BatchNormalization()(x)
        activation = kwargs.get('activation')
        if not activation: 
            x = LeakyReLU(alpha=0.1)(x) #默认使用LeakyReLU
        else:
            print('#使用:', activation )
            x = Activation(activation)(x)
        return x

    @showinfo
    def __res_blocks(self, x, filters, num_blocks, name):
        """
        @ A series of resblocks starting with a downsampling Convolution2D
        :param
            x: input 
            filters: 卷积核数量
            num_blocks: 残差block数量
        :return 
            Output tensor  
        """ 
        # uses left and top padding instead of 'same' mode
        x = ZeroPadding2D(((1,0),(1,0)))(x)
        #图片尺寸 缩小一半
        x = self.__Conv2D_BN(x, filters, (3,3), strides=(2,2))
        #多个残差块
        for i in range(num_blocks):
            with tf.variable_scope(name+str(i)):
                y = self.__Conv2D_BN(x, filters//2, (1,1))
                y = self.__Conv2D_BN(y, filters//2, (3,3))
                y = self.__Conv2D(y, filters, (1,1), use_bias=False)
                y = BatchNormalization()(y)
                x = Add()([x,y])  #residual shoutcut
                x = LeakyReLU()(x)  #激活放Add后
        return x

        
    def __inception_block(self, x, filters):
        """
        :param
            x: input 
            filters: 输入也是输出的卷积核数量
        :return
            concat[1/6, 1/6, 1/6, 1/6, 1/6, 1/6] 同样尺寸
        """
        a = AveragePooling2D(3, 1, padding='same')(x)
        a = self.__Conv2D(a, filters//6, (1,1))
        b = self.__Conv2D(x, filters//6, (1,1))
        c = self.__Conv2D(x, filters//3, (1,1))
        c1 = self.__Conv2D(c, filters//6, (1,3))
        c2 = self.__Conv2D(c, filters//6, (3,1))
        d = self.__Conv2D(x, filters//3, (1,1))
        d = self.__Conv2D(d, filters//3, (1,3))
        d = self.__Conv2D(d, filters//3, (3,1))
        d1 = self.__Conv2D(d, filters//6, (1,3))
        d2 = self.__Conv2D(d, filters//6, (3,1))
        return tf.concat([a,b,c1,c2,d1,d2], axis=-1)


    """
    darknet骨干网络
    """
    def darknet_body(self, input):
        """
        @@@ Darknent body having 52 Convolution2D layers
        :param
            input: input tensor
        :return 
            Output tensor  
        """         
        x = self.__Conv2D_BN(input, 32, (3,3))
        x = self.__res_blocks(x, 64, 1, 'Residual_Block_1_')
        x = self.__res_blocks(x, 128, 2, 'Residual_Block_2_')
        x = self.__res_blocks(x, 256, 8, 'Residual_Block_3_')
        x = self.__res_blocks(x, 512, 8, 'Residual_Block_4_')
        x = self.__res_blocks(x, 1024, 4, 'Residual_Block_5_')
        return x    


    """
    new net 骨干网络
    """
    def newnet_body(self, input):
        """
        @ new
        :Args
            input: input tensor
        :returns 
            Output tensor  
        """         
        x = self.__Conv2D_BN(input, 16, (3,3))
        x = self.__res_blocks(x, 32, 1, 'Residual_Block_1_')
        x = self.__res_blocks(x, 48, 2, 'Residual_Block_2_')
        x = self.__res_blocks(x, 64, 4, 'Residual_Block_3_')
        x = self.__res_blocks(x, 92, 4, 'Residual_Block_4_')
        y = self.__res_blocks(x, 128, 2, 'Residual_Block_5_')
        x = self.__res_blocks(y, 164, 2, 'Residual_Block_5_')
        x = self.__res_blocks(x, 192, 1, 'Residual_Block_5_')

        x = UpSampling2D(2)(x)  #[?,6,6,192]
        y = self.__Conv2D(y, 128, (3,3), strides=(2,2))  #[?,6,6,128]
        return Concatenate()([x,y])   

    def __make_last_layer(self, input, filters, out_filters):
        """
        @ 6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer
        :param
            input: input tensor
            filters: 卷积核数目
            out_filters: y输出卷积核数目（通道数）
        :return 
            x, y: 两个Output tensor  
        """ 
        x = self.__Conv2D_BN(input, filters, (1,1))
        x = self.__Conv2D_BN(x, filters*2, (3,3))
        x = self.__Conv2D_BN(x, filters, (1,1))
        x = self.__Conv2D_BN(x, filters*2, (3,3))
        x = self.__Conv2D_BN(x, filters, (1,1))

        y = self.__Conv2D_BN(x, filters*2, (3,3))
        y = self.__Conv2D(y, out_filters, (1,1))
        return x, y

    def fpn(self, net, *args, **kwargs):
        """
        @ feature pyramid networks 特征金字塔网络FPN
        #同时利用低层特征高分辨率和高层特征的高语义信息，通过融合这些不同层的特征达到预测的效果。
        #并且预测是在每个融合后的特征层上单独进行的
        :param
            net: 骨干网络model
        :return 
            [Output-tensor,] 效果：所有尺度下的特征都有丰富的语义信息 
        """  
        #以每个‘stage’为一个pyramid level，取每个stage最后layer输出的feature map作为pyramid level
        #对应darknet,是stage2 stage3 stage4 stage5的res block各自最后一个输出
        #stage2 stage3 stage4 stage5的res block各自最后一个输出
        darknet = net  #input[416,416]
        C5 = darknet.layers[184].output  #stage5输出 [13,13,1024]
        C4 = darknet.layers[152].output  #stage4输出 [26,26,512]
        C3 = darknet.layers[92].output  #stage3输出 [52,52,256]
        C2 = darknet.layers[32].output  #stage2输出 [104,104,128]

        # P2，P3，P4，P5
        out_filters = 256  #输出通道数 都一样
        P5 = self.__Conv2D(C5, out_filters, (1,1), name='fpn_p5')
        P4 = Add()([  #横向连接
            UpSampling2D(2)(P5),  #上采样跟C4同尺寸
            self.__Conv2D(C4, out_filters, (1,1)),  #1*1卷积跟P5通道数一样
            ])
        P3 = Add()([   
            UpSampling2D(2)(P4),
            self.__Conv2D(C3, out_filters, (1,1)),
            ])
        P2 = Add()([   
            UpSampling2D(2)(P3),
            self.__Conv2D(C2, out_filters, (1,1)),
            ])
        # P2-P4最后又做了一次3*3的卷积，作用是消除上采样带来的混叠效应
        P4 = self.__Conv2D(P4, out_filters, (3,3), name='fpn_p4')
        P3 = self.__Conv2D(P3, out_filters, (3,3), name='fpn_p3')
        P2 = self.__Conv2D(P2, out_filters, (3,3), name='fpn_p2')
        # 最后得到了融合了不同层级特征的特征图列表
        rpn_feature_maps = [P2, P3, P4, P5]
        return rpn_feature_maps


    def create_model(self):
        """
        @ 创建 模型
        :param
            None 
        :return 
            model
        """
        model_input = Input(shape=X_SHAPE) 
        #new网络模型
        newnet = Model(model_input, self.newnet_body(model_input)) 
        # darknet.load_weights('lib/darknet53.h5')
        # P2, P3, P4, P5 = self.fpn(darknet)
        
        # for layer in darknet.layers[:]:
        #     layer.trainable = False  #darknet部分 权重不训练  
        x = self.__Conv2D_BN(newnet.output, 160, (1,1))    
        x = self.__Conv2D_BN(newnet.output, 80, (1,1))    
        x = MaxPool2D(2)(x)
        x = Flatten()(x)
        predictions = Dense(CLASS_NUM, activation='softmax', name='softmax1')(x)
        # 构建我们需要训练的完整模型
        model = Model(model_input, predictions)
        model.load_weights('backup/weights-1244221-27-0.386.hdf5', by_name=True)


        # 编译模型（一定要在锁层以后操作）
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=4e-3), #微调时考虑 (lr=1e-4)
               loss='categorical_crossentropy',
               metrics=['acc'],
               )            
        return model

    def create_model_1(self):
        """
        @ 创建 模型
        :param
            None 
        :return 
            model
        """
        model_input = Input(shape=X_SHAPE) 
        #darknet 53成网络模型
        darknet = Model(model_input, self.darknet_body(model_input)) 
        #加载darknet 预训练参数 （待验证）
        darknet.load_weights('lib/darknet53.h5')
        for layer in darknet.layers[:]:
            layer.trainable = False  #darknet部分 权重不训练
        # print(darknet.summary())

        x = self.__Conv2D_BN(darknet.output, 256, (1,1))
        x = self.__res_blocks(x, 256, 2, name='Residual_Block_6_')
        x = self.__res_blocks(x, 128, 1, name='Residual_Block_7_')

        x = UpSampling2D(2)(x)
        y = MaxPool2D()(darknet.output)
        x = Concatenate()([x, y])

        new_darknet = Model(model_input, x)

        # P2, P3, P4, P5 = self.fpn(new_darknet)
        # print(P2, P3, P4, P5)

        # x = self.__Conv2D_BN(P4, 128, (1,1))
        # x = AveragePooling2D()(x)
        x = Flatten()(new_darknet.output)

        predictions = Dense(CLASS_NUM, activation='softmax')(x)
        # 构建我们需要训练的完整模型
        model = Model(model_input, predictions)
        # model.load_weights('backup/weights-1244221-04-0.283.hdf5', by_name=True)


        # 编译模型（一定要在锁层以后操作）
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=4e-3),
               loss='categorical_crossentropy',
               metrics=['acc'],
               )            
        return model

    def create_model_vgg16_base(self):
        para = Pre_Model().vgg16_para
        model_input = Input(shape=para[1]) 
        #vgg16  第五 第四 第三层输出
#         x5, x4, x3 = self.seperate_pre_model_tensor('vgg16', model_input)
#         print(x5, x4, x3)
         
        x = Flatten()(model_input)   
        predictions = Dense(CLASS_NUM, activation='softmax', name='softmax1')(x)
        model = Model(model_input, predictions)
#         model.load_weights('backup/weights-1244221-27-0.386.hdf5', by_name=True)
        # 编译模型 
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=4e-3), #微调时考虑 (lr=1e-4)
               loss='categorical_crossentropy',
               metrics=['acc'],
               )            
        return model           
        
        
    def create_model_2(self):
        """
        @ 创建 模型
        :param
            None 
        :return 
            model
        """
        model_input = Input(shape=X_SHAPE) 
        #先加载预定义的模型
#         base_model = vgg16.VGG16(include_top=False,
#                         weights='imagenet',
#                         input_tensor=model_input,
#                         )
        base_model = resnet50.ResNet50(
                        include_top=False,
                        weights='imagenet',
                        input_tensor=model_input,
                        )
        for layer in base_model.layers[:]:
            layer.trainable = False  #darknet部分 权重不训练
        base_model.summary()
        x = base_model.output
            
        x = DepthwiseConv2D((3,3), strides=(2,2), padding='same')(x)
        x = DepthwiseConv2D((3,3), strides=(2,2))(x)
        x = DepthwiseConv2D((3,3))(x)
#         x = self.__Conv2D_BN(x, 512, (1,1))
#         x = self.__inception_block(x, 256)  #尺寸不变  
        #vgg block4的最后层输出  pool后是(None, 26, 26, 512)
#         y = vgg.get_layer('block4_pool').output  

#         x = self.__Conv2D_BN(base_model.output, 256, (3,3), strides=(2,2))
        x = Flatten()(x)   
        predictions = Dense(CLASS_NUM, activation='softmax', name='softmax1')(x)
        model = Model(model_input, predictions)
#         model.load_weights('backup/weights-1244221-27-0.386.hdf5', by_name=True)
        # 编译模型 
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=4e-3), #微调时考虑 (lr=1e-4)
               loss='categorical_crossentropy',
               metrics=['acc'],
               )            
        return model        

    def create_model_easy(self):
        """
        @ 创建 简单的模型
        :param
            None 
        :return 
            model
        """
        def _reduction_block(x, filters):
            """@inception 尺寸减少一半 通道增加一倍"""
            y = MaxPool2D(3, 2, padding='same')(x)   #3X3 strides 2
            y = self.__Conv2D(y, filters//2, (1,1))
            z = self.__Conv2D(x, filters//2, (1,1))
            z = self.__Conv2D(x, filters, (3,3), strides=(2,2), padding='same') #这个权重最大
            x = self.__Conv2D(x, filters//4, (1,1))
            x = self.__Conv2D(x, filters//4, (1,7))
            x = self.__Conv2D(x, filters//4, (7,1))
            x = self.__Conv2D(x, filters//2, (3,3), strides=(2,2), padding='same')
            x = tf.concat([x, y, z], axis=-1) 
            return x

        model_input = Input(shape=X_SHAPE) 
        x = self.__Conv2D_BN(model_input, 32, (3,3), strides=(2,2), padding='same')
        x = self.__Conv2D_BN(x, 32, (3,3), padding='same')
        
        x = self.__res_blocks(x, 48, 1, name='Residual_Block_1_')  #(None, 104, 104, 48)
        x = self.__res_blocks(x, 64, 2, name='Residual_Block_2_')  #(None, 52, 52, 64)
        
        x = _reduction_block(x, 48)  #通道增加1倍 (None, 26, 26, 96)
        x = self.__inception_block(x, 96)  #尺寸不变 (None, 26, 26, 96)
        
        x = self.__res_blocks(x, 128, 1, name='Residual_Block_3_')
        
        y = MaxPool2D(3, 2, padding='same')(x)  #3X3 strides 2
        y = self.__Conv2D_BN(y, 64, (1,1))
        x = self.__Conv2D_BN(x, 64, (3,3), strides=(2,2), padding='same')
        x = self.__Conv2D_BN(x, 128, (1,1))
        x = Concatenate()([x, y])  #(None, 7, 7, 192)
 
#         x = self.__Conv2D_BN(x, 64, (3,3), strides=(2,2))
        x = AveragePooling2D(5, 2)(x)
        x = Flatten()(x)

        predictions = Dense(CLASS_NUM, activation='softmax')(x)
        # 构建我们需要训练的完整模型
        model = Model(model_input, predictions)
        # model.load_weights('backup/weights-1244221-04-0.283.hdf5', by_name=True)


        # 编译模型（一定要在锁层以后操作）
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-3),
               loss='categorical_crossentropy',
               metrics=['acc'],
               )            
        return model
                
    def load_weights(self, weights, by_name=True):
        """
        @ model加载weights
        :param
            weights: weights文件地址
            by_name: 是否按op名覆盖
        :return 
         """          
        self.model.load_weights(weights, by_name=by_name)

    def fit(self, xs, vs, ratio=1):
        """
        @@@ model fit
        :param
            xs: 训练集
            vs: 验证集
            ratio: 比例0-1 pc上比较慢 可以缩小epoch步数
        :return 
            history 对象。其 History.history 属性是连续 epoch 训练损失和评估值，以及验证集损失和评估值的记录（如果适用）
        """    
        print('model 开始 fit... 取{:.0%}的xs&vs数据'.format(ratio))
        # 当评价指标不在提升时，减少学习率
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)  
        #停止阈值min_delta和patience需要相互配合，避免模型停止在抖动的过程中。min_delta降低，patience减少；而min_delta增加，则patience增加。
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
        #checkpoint保存
        checkpoint = ModelCheckpoint(filepath= 'backup/weights-vgg16-{epoch:02d}-{val_acc:.3f}.hdf5', #文件路径
                                verbose= 1,
                                save_weights_only = True,
                                mode = 'max',
#                                 load_weights_on_restart = True,
                                period = 3,
                                )
        hist = self.model.fit(xs,
                    epochs= EPOCHS,
                    verbose= 1,
                    steps_per_epoch= int(STEP_PER_EPOCH['xs']*ratio),
                    validation_data= vs,
                    validation_steps= int(STEP_PER_EPOCH['vs']*ratio),
#                     callbacks= [early_stopping, checkpoint, reduce_lr],  
                    )
        self.model.save_weights('my_model_weights.h5') #最后保存一次
        return hist

    def evaluate(self, vs):
        """
        @ model 测试下 验证集
        :param
            vs: input data
        :return 
             
        """ 
#         print(STEP_PER_EPOCH['vs'])
        self.model.evaluate(vs, 
                verbose= 1,
                steps= STEP_PER_EPOCH['vs'],
                )
    
    def predict(self, x):
        """
        @ predict
        :param
            x: input data
        :return 
            result: [[]] (batch,classNum) 
        """ 
        result = self.model.predict(x,
                    # verbose=1,
                    )
        return result        
                
    def pre_predict(self, files, weights=None):
        """
        @ model predict预处理图片
        :param
            files: 传入的图片文件名
            weights: model weights path
        :return 
            result_df: FileName type的dataframe 
        """ 
        pass
    
    def test_pre_model(self, dataset):
        """@测试 预训练模型 从tensorflow读出来的数据 能不能用来训练"""
        model = self.model
        print(model)
    
    @timeit
    def predict_test(self, ratio=1, weights=None):
        """
        @ model predict 测试集，并返回测试结果
        :param
            ratio: 比例0-1 取多少数据 0.1=10%
        :return 
            result_df: FileName type的dataframe 
        """ 
        def _get_data(f):
            #获取并处理图片
            img = Pic.read_image(f)
            img = Pic.preprocessing(img, region='big')[0] #test图片未处理 裁多一点
            return img
        
        Flag = False  #是否有文件不存在
        result = {}  #保存预测结果
        df = Pic.get_test_NLs(ratio)  #读取全部test图片
        flist = df['FileName'] 
        #检查图片是否都存在
        for f in flist:
            if not os.path.isfile(f):
                print(f, ' 图片不存在！')
                Flag = True
                continue         
        if Flag:
            print('错误，请前检查文件')
            return
        else:
            print('所有{}张图片都存在，检查完毕。'.format(len(flist)))
        
        if weights:
            print('predict 加载weights: {}'.format(weights))
            self.model.load_weights(weights)
        
        if not isinstance(flist, np.ndarray):  #传入是不是ndarray
            try:
                flist = np.array(flist)  
            except Exception as e:
                print('无法转换成np.array')
                raise e        
        
        batchsize= 8
        rank = 3 #取概率最大 排名前3的
        for i in range(0, len(flist), batchsize):
            x = np.array([_get_data(f) for f in flist[i:i+batchsize]])
            ret = self.predict(x)
            for j in range(len(ret)):
                if (i+j) % 250 == 0:
                    print('已predict图片{}张...'.format(i+j))
                _ret = ret[j]
                index = np.argsort(-_ret)[:rank]
                fname = flist[i+j][8:]  #图片名（去除路径后）
#                 print('{}: {} '.format(i+j, fname)) 
#                 print('预测分类{} 概率{}'.format(index, _ret[index]))
                result[i+j] = {'FileName':fname, 'type':index[0]}
        df = pd.DataFrame.from_dict(result)
        return df.T  #'FileName' 'type'到索引去了 转置一下行列
        
def _show_weights(id):
    """@显示weights
    :param
        id: 第几层        
    """
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    weights = m.model.layers[id].weights
    print(sess.run(weights[0][...,:3]))
        
def _show_h5():
    import h5py
    f = h5py.File('lib/darknet53.h5', 'r') 
    # it = f['/model_weights/add_1'] 
    # it = f.get('/model_weights/batch_normalization_1/batch_normalization_1') 
    it = f.get('/model_weights/conv2d_1/conv2d_1').get('kernel:0')
    # [i for i in it.items()]
    # for i in ['beta:0', 'gamma:0', 'moving_mean:0', 'moving_variance:0']:
    #     print(it.get(i).shape, it.get(i).value)
    print(it.value[...,:3])
    
def show_h5_list():
    """@显示h5文件的内容"""
    #遍历文件中的一级组
    print(list([key for key in f.keys()]))
    for group in f.attrs():
        # print (group)
        #根据一级组名获得其下面的组
        group_read = f[group]
        #遍历该一级组下面的子组
        for subgroup in [k for k in group_read.keys()]:
            print(subgroup)
            #根据一级组和二级组名获取其下面的dataset          
            dset_read = f[group+'/'+subgroup] 
            #获取dataset数据
            #遍历该子组下所有的dataset
            for dset in dset_read.keys():
                _dataset = f[group+'/'+subgroup+'/'+dset]
                print(_dataset.name)
                data = np.array(_dataset)
                print(data.shape)    

def show_predict(model, df, num):
    """
    @ 测试 显示predict结果
    :Args
        model: 使用的model
        df: 文件名dataframe
    """
    i, l = 0, 0
    type = []  #存放检查出的类别
    for index, row in df[:num].iterrows():
        path = row.loc['FileName']
        t = int(row.loc['Code'])  #图片所属类别
        if not os.path.isfile(path):
            print(f, '文件不存在！')
            continue
#                  
        img = Pic.read_image(path)
        img = Pic.preprocessing(img, region='small')[0]
        if isinstance(img, np.ndarray):
            if len(img.shape) == 3:
                img = np.expand_dims(img, 0)
            # print((img[0,:3,:3,0]))
#             
        result = model.predict(img)[0]
        r = np.argsort(-result)[:3]
        print('y_ture:{}  y:{}'.format(t, r))
        if t == result.argmax():
            i+=1
            print('======={}'.format(result[r]))
        elif t in r:  #在前三里面
            l += 1
            print('-------{}'.format(result[r]))
        else:
            print('{}'.format(result[r]))
        
        type.append(r[0])
            #查看 是什么样的图片 predict不对
#                 img = Pic.read_image(path)
#                 plt.imshow(img)   
#                 plt.show()
    import collections
    r = collections.Counter(type)  #统计词频
    print('result: 1st [{:.0%}] -3rd [{:.0%}]'.format(i/num, (i+l)/num))
    print(len(r), r)
                
                    
if __name__ == '__main__':
    dg = DatasetGenerator('vgg16', augmentation=1)
#     dg._test_read_write()
 
    XS_dataset = dg.read_data_from_TFRecoard('xs') #单label训练集
    VS_dataset = dg.read_data_from_TFRecoard('vs')
#     
#     iterator = XS_dataset.make_one_shot_iterator()
#     t, l  = iterator.get_next()
#     print('dddddddd', t.shape, l.shape)
#     
    m = MyModel()
#     m.model.summary()

#     m.test_pre_model(XS_dataset)
#     print('预测test数据集的结果：')
#     weights = 'backup/weights-easy-17-0.227.hdf5'
#     ret = m.predict_test(ratio=1, weights=weights)
#     print(ret['type'].drop_duplicates())
#     print(ret)
#     ret.to_csv('lib/submit.csv', index=False)  #生成csv文件
    
#     m.load_weights(weights)
    m.fit(XS_dataset, VS_dataset, ratio=0.1)
#     m.evaluate(XS_dataset)
    
    
    # img, label = dg.iterator_from_tfrecoard('xs') #测试用    
    
#     t = m.model.evaluate(VS_dataset, verbose=1, steps=STEP_PER_EPOCH['vs'],)
    
#     print('在训练集上predict 验证一下：')
#     df = Pic.get_train_NLs('single')
#     show_predict(m, df, 100)  


    def move_multi_img():
        """@把multi label图片单独移到一个文件夹中"""
        dg = DatasetGenerator()
        #     print(STEP_PER_EPOCH)
        l = pd.concat([dg.x_multi_NLs,dg.v_multi_NLs], axis=0)
        l = l['FileName'].tolist()
        dir = r'D:/eclipse/workspace/clound/Train/'
        import shutil
        dst = dir+'mutil'   #目标文件夹
        for f in l[:]:
            src = os.path.join(dir+f[9:])
            if not os.path.isfile(src): 
                print('no ', src)
                continue
            try:
    #             img = Pic.read_image(src)
        #         print(img)
                shutil.copy(src, dst) 
                os.remove(src)
            except:
                print('error ', src)
                continue            