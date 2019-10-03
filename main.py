"""****************************************"""
from matplotlib.pyplot import subplot
""" 处理图片                 """
"""****************************************"""
import tensorflow as tf
from tensorflow.train import Features, Feature, Example, Int64List, FloatList, BytesList
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import pandas as pd
import os

TRAIN_IMG_DIR = 'trainImg/' #训练图片的目录
TEST_IMG_DIR = 'testImg/'  #测试图片的目录
TRAIN_LIST_FILE = 'Train_label.csv' #训练图片文件名和label的列表文件
X_SHAPE = (500, 500, 3) # 预处理后图像的大小

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
            print('[path] error...')
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
            if (i+1)%num == 0: 
                plt.show()            
        
    @staticmethod
    def preprocessing(img):
        """
        @@@ 对图像数据进行预处理
        :param
            img: Image格式的图片
        :return
            norm后的array [H,W,3]，值范围[-0.5,-0.5]
        """
        #其他处理。。。

        #resize crop变成H*W大小
        img = Pic._resize(img)
        #PIL Image转换为array
        img_array = np.asarray(img, np.float32)

        # img_norm = tf.image.per_image_standardization(timg) #tensorflow中对图像标准化预处理的API
        img_norm = img_array/255. - 0.5  #[-0.5,0.5]归一化
        return img_norm

    @staticmethod
    def _resize(img):
        """
        @@@ 对图像裁切 缩减
        :param
            img: PIL.Image
        :return
            PIL.Image 预定义大小的图片 
        """
        H, W = X_SHAPE[:2]   #输入模型需要的图片尺寸
        h, w = img.size[::-1]  #原始图片的高 宽
        new_size = None #用来缩放的新尺寸
        if h > w:  #如果原图宽比较小
            if w > W*1.1:  #宽比要求的尺寸高10%以上
                new_size = (int(W*1.1), int(h/w*W*1.1)) #宽缩到110% 剩下的下一步再裁剪掉
            elif w < W:  #宽比要求的尺寸小
                new_size = (W, int(h*W/w))  #宽放大到需要尺寸 高同步放大
            else: image = img
        else: #原图高比较小
            if h > H*1.15:  #高 多15%以上
                new_size = (int(w/h*H*1.15), int(H*1.15))
            elif h < H:
                new_size = (int(w*H/h), H)
            else: image = img
            
        if new_size: #改变图片大小
                image = img.resize(new_size, Image.ANTIALIAS) #压缩质量还不行 要改 
                
        #对resize好的图片 进行裁剪，满足预定义尺寸H,W （根据天空的实际图片确定的策略）
        h, w = image.size[::-1] #resize后的尺寸
        top = int(0.2 * (h-H)) #上高裁切 取 多余的20%
        left = int(0.5 * (w-W)) #宽度多余的 左右各裁剪一半
        image = image.crop((left, top, left+W, top+H)) #四周裁剪 留下H*W区域
        return image

    @staticmethod
    def get_train_NLs(lkind, ratio=None):
        """
        @@@ 获取train所有的image的列表(name,label)
        :param
            lkind: 'single'单label, 'multi'多label 
        :return 
            dataframe(shuffled)
        """
        df = pd.read_csv(TRAIN_LIST_FILE)
        if lkind == 'single':  #获取单label的(图片名和label列表)
            NLs = df[df.apply(lambda x: len(x['Code'])<=2, axis=1)].copy()
        elif lkind == 'multi':  #多label  
            NLs = df[df.apply(lambda x: len(x['Code'])>2, axis=1)].copy()
        else:
            raise RuntimeError('输入错误 lkind= "{}"'.format(lkind) )
        #文件名上添加路径
        NLs['FileName'] = NLs['FileName'].apply(lambda x: TRAIN_IMG_DIR + x)    
        #打乱顺序 frac是要返回的比例 1=100%
        NLs = NLs.sample(frac=1).reset_index(drop=True) 
        if ratio:  #为了应付colab经常删文件，故缩小样本节约时间,正常不要这样操作
            return NLs[:int(ratio * len(NLs))]
        else: 
            return NLs

    @staticmethod
    def get_class_num():
        """
        @@@ 获取单标签的分类总数 也就是最后层神经元数目
        :return
            int
        """
        NLs = Pic.get_train_NLs('single')
        return len(NLs['Code'].unique())
    
    
"""****************************************"""
""" 处理数据                 """
"""****************************************"""
NUM_PER_TFRECOARD = 2000 #每个tfrecoard文件包括的图片数
CLASS_NUM = Pic.get_class_num()  #单标签 标签类别数目
TFR_DIR = 'lib/'  #tfrecoard文件的存放路径
#用colab时 每次都重新生成文件 可以考虑缩小样本 0.1=1/10 None=不缩小
COLAB_PIC_RATIO = 1 
BATCH_SIZE = 20  #批量读取 1bacth多少张照片
LABEL_CLASS = {  #train数据集的分类
            'xs' : '单label训练集',
            'vs' : '单label验证集',
            'xm' : '多label训练集',           
            'vm' : '多label验证集',    
    }

class DatasetGenerator:
    def __init__(self):
        #单label训练集和验证集的文件列表
        self.x_single_NLs, self.v_single_NLs = self.__separete(Pic.get_train_NLs('single', COLAB_PIC_RATIO))
        #多label训练集和验证集的文件列表
        self.x_multi_NLs, self.v_multi_NLs = self.__separete(Pic.get_train_NLs('multi', COLAB_PIC_RATIO))

        #检查文件夹的tfrecoard文件，若没有则生成新的
        self.check_tfrecoard()
        
    def check_tfrecoard(self):
        """
        @@@ 检查文件夹的tfrecoard文件，若没有则生成新的
        :param
            None
        :return 
            None
        """       
        #类别名  单label训练集和验证集   多label训练集和验证集 
        classes = ['xs', 'vs', 'xm', 'vm']
        #检查每一个数据集的tfrecoards是否存在
        for c in classes:
            fnames = self.__get_tfr_file_names(lclass=c) #对应的文件名
#             print('{}检查 TFRecoard:{}'.format(c, fnames))
            for f in fnames:
                if not os.path.isfile(f):
                    #只要有一个文件不存在，就全部重新生成
                    print('{}不存在，"{}"所有tfrecoard重新生成[{}个*{}张/个]...'.format(
                        f, LABEL_CLASS[c], len(fnames), NUM_PER_TFRECOARD))
                    if self._create_class_tfrecoards(c):
                        break #全部重新生成 后面的不用检查了
                    else:
                        print('  创建失败！')
            print('对{} "{}"的{}个TFRecoard 检查完毕！'.format(c, LABEL_CLASS[c], len(fnames)))
        print('OK! "{}"的所有TFRecoard文件检查完毕！\n'.format(classes))
        
    def _create_class_tfrecoards(self, lclass):
        """
        @@@ 把当前数据集图片列表 生成tfrecoard文件 
        :param
            lclass: train数据集的分类
        :return 
            True成功  False失败
        """     
        assert lclass in LABEL_CLASS.keys(), "lcalss 输入error..."
        #当前数据集下的 图片列表
        df = self.__get_NLs_by_class(lclass)
        #应该产生的文件名
        fnames = self.__get_tfr_file_names(lclass=lclass)
        print('{} {}张图片 准备create TFRecoard:{}'.format(lclass, len(df), fnames))
        
        start = 0
        i = 0 #文件名序号
        while start < len(df):
            fname = fnames[i]  #当前写入的文件名
            df_slice = df[start:start+NUM_PER_TFRECOARD]
            if self.__create_tfrecoard(fname, df_slice):
                print('create TFRecoard[{}]  从df[{}:]... success'.format(fname, start))
                i += 1
                start += NUM_PER_TFRECOARD
            else:
                print('create TFRecoard[{}]  从df[{}:]... faile'.format(fname, start))
                return false

        return True        
        
    def __create_tfrecoard(self, tfrname, df, option=None):
        """
        @@@ 把输入的图片列表 生成一个tfrecoard文件  若存在则覆盖
        :param
            tfrname：生成的tfrecoard文件 名
            df：要保存的图片 FileName和lable的列表
        :return 
            True成功  False失败
        """            
        print('  准备创建tfrecoard文件 【{}】...'.format(tfrname, len(df)))
        ok_num = 0  #成功写入tfrecoard的图片数目
        trf_writer = tf.io.TFRecordWriter(tfrname, options=option)
        for index, (fname, label) in df.iterrows():
            try:
                img = Pic.read_image(fname)
            except:
                print('  {}不存在，跳过...'.format(fname))
                continue  #文件不存在 忽略 处理下一个
            img = Pic.preprocessing(img) #裁剪 归一化等预处理
            x_shape = img.shape #处理后的图片尺寸 等于输入到模型中的图片尺寸
            assert x_shape ==X_SHAPE, "准备写入TRF的图片尺寸不等于{}".format(X_SHAPE)
            img = img.reshape(-1) #变成一维
#             img_list = img.tolist()  
            label = [int(x) for x in label.split(';')]
            #对多label的情况，把label固定到5个长度（batch要统一长度）  即一张图片最多5种云
            if len(label) > 1:
                label += [0,0,0]
                label = label[:5]
            name = [tf.compat.as_bytes(fname)]
            #内层feature编码方式
#             print('内层feature编码', name, label)
            feature_internal = {
                        'image_raw' : Feature(float_list = FloatList(value=img)), #图形数据
                        'name'  : Feature(bytes_list = BytesList(value=name)), #路径文件名
                        'img_shape' : Feature(int64_list = Int64List(value=x_shape)),  
                        'label'   : Feature(int64_list = Int64List(value=label)),  #字符串的label   
                        }
            #使用tf.train.Example将features编码数据封装成特定的PB协议格式
            example = Example(features=Features(feature=feature_internal))
            #将序列化为字符串的example数据写入协议缓冲区
            trf_writer.write(example.SerializeToString())
            ok_num += 1
        print('  successfully created tfrecoard文件 【{}】, 共图片{}个...'.format(tfrname, ok_num))
        #关闭TFRecords文件操作接口    
        trf_writer.close()  
        return True
        
    def read_data_from_TFRecoard(self, lclass, batch_size=BATCH_SIZE):
        """
        @@@ 获取数据集数据 从tfrecoard里读取
        :param
            lclass: train数据集的分类
        :return 
            interator 可迭代的image label数据
            [batch,H,W,3] if batch， 3D[H,W,3] if no batch
            None if no file to read
        """
        assert lclass in LABEL_CLASS.keys(), "lcalss 输入error..."
        fnames = self.__get_tfr_file_names(lclass=lclass)
        
        tfr_files = [] #待读取的tfrecoard文件列表
        for f in fnames:
            if os.path.exists(f): tfr_files.append(f)
        if tfr_files == []: return None
          
        print('从[{}...]开始读取{}个TFRecoard...'.format(tfr_files[0], len(tfr_files)))
        dataset_raw = tf.data.TFRecordDataset(tfr_files)
  
        # Set the number of datapoints you want to load and shuffle 
        dataset = dataset_raw.shuffle(buffer_size = NUM_PER_TFRECOARD//2)
        #执行解析函数 得到数据集    
        dataset = dataset.map(self.__parse_function)
        # 不加参数=无限重复数据集
        dataset = dataset.repeat()                   
        # 定义batchsize大小
        if batch_size: #batch时 必须保证各元素长度都一样
            dataset = dataset.batch(batch_size)

#         # Create an iterator
#         iterator = dataset.make_one_shot_iterator()
#         return iterator
        return dataset
        
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
        features = {
            'image_raw' : tf.io.VarLenFeature(dtype=tf.float32),
            'name' : tf.io.VarLenFeature(dtype=tf.string),
            'img_shape' : tf.io.FixedLenFeature(shape=(3,), dtype=tf.int64),
            'label' : tf.io.VarLenFeature(dtype=tf.int64)
            }
        # 把序列化样本和解析字典送入函数里得到解析的样本
        parsed_example = tf.io.parse_single_example(example_proto, features) #返回字典
        # 解码 
        img = tf.sparse.to_dense(parsed_example['image_raw'], default_value=0) # 稀疏->密集表示
        img = tf.reshape(img, parsed_example['img_shape']) # 转换为原来形状
        label = self.__deal_label(parsed_example['label'])
        name = tf.sparse.to_dense(parsed_example['name'], default_value='')
        name = tf.squeeze(name,)
        return img, name, label   #如果batch 则必须保证各元素长度都一样     

    def __deal_label(self, label):
        """
        @@@ 把原始字符串的label 变成int的list
        :param
            label：原始label 单label 或者多label(e.g. 1:12)
        :return 
            list[int]  
        """
        label = tf.sparse.to_dense(label, default_value=0)
        return label       

    def __separete(self, df):
        """
        @@@ 把数据集分割成训练集和验证集 
        :return
            x,v 按比例划分的训练集和验证集
        """
        ratio = 0.7  #训练集的大小 70%
        truncation = int(len(df) * ratio) #训练集和验证集分割点
        x = df[:truncation] #训练集
        v = df[truncation:] #验证集
        return x,v
    
    def __get_NLs_by_class(self, lclass):
        """
        @@@ 通过lclass 获取对应数据集的图片 FileName和lable的列表
        :param
            lclass: 'xs' 'vs' 单label训练集和验证集 
                'xm' 'vm' 多label训练集和验证集 
        :return
            dataframe
        """
        xs = self.x_single_NLs
        vs = self.v_single_NLs
        xm = self.x_multi_NLs
        vm = self.v_multi_NLs
        try:
            df = eval(lclass)
        except:
            raise RuntimeError('输入错误 lclass= "{}"'.format(lclass))
        return df.reset_index(drop=True)

    def __get_tfr_file_names(self, lclass):
        """
        @@@ 获取各种数据集应该产生的tfrecoard文件的列表
        :param
            lclass: 'xs' 'vs' 单label训练集和验证集 
                'xm' 'vm' 多label训练集和验证集 
        :return 
            list tfrecoard文件名列表
        """
        df = self.__get_NLs_by_class(lclass) #图片名列表
        file_num = (len(df)-1) // NUM_PER_TFRECOARD + 1

        #设置的文件名 (考虑加路径 暂时同一目录下)
        names = [TFR_DIR + lclass + '_' + str(x+1) + '.tfrecord' for x in range(file_num)]
        return names
    
    def __get_clound_name(self, labels):
        """
        @@@ 根据label 获取对应云的名字
        :param
            labels: int的list
        :return 
            list 云的名字
        """        
        clound = {    #编号（type）  云状类型
                0:'', #仅填充用
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
        names = []
        for label in labels:
            names.append(clound[label])
        return names

    def _test_get_img(self, lclass, num=1):
        """
        @ 测试用 获取一批图片
        :param
            lclass: 'xs' 'vs' 单label训练集和验证集 
                'xm' 'vm' 多label训练集和验证集
            num: 要几个
        :return 
            list
        """
        df = self.__get_NLs_by_class(lclass=lclass)[:num]
#         print('test取到的图片数据：', df)
        return df

    def _test_read_write(self):
        """@测试整个流程"""
        sess =  tf.compat.v1.InteractiveSession() 
        sess.run(tf.compat.v1.global_variables_initializer())
        
        classes = ['xs', 'vs', 'xm', 'vm'][2:]
        for c in classes:
            print('测试{} "{}"tfrecoard...单文件有{}张图片  BATCH：{}'.format(
                c, LABEL_CLASS[c], NUM_PER_TFRECOARD, BATCH_SIZE))
            dataset = dg.read_data_from_TFRecoard(c)
            self.__test_print(dataset)
#         print(dataset.__iter__().get_next())
#

    def __test_print(self, dataset): 
        """@测试 显示iterator的元素"""         
        iter = tf.compat.v1.data.make_one_shot_iterator(dataset)
        imgs, names, labels = iter.get_next()
        p_freq = {} #打印频率 
        try:
            for i in range(55):
                image = imgs.eval()+0.5
                name = names.eval()
                label = labels.eval()
                print('  batch {} shape:{} '.format(i+1, image.shape))
                for j in range(image.shape[0])[:]:
#                     p_freq[name[j]] = p_freq.get(name[j], default=0) + 1
                    print('  No.{} image name:{}  label:{} '.format(j+1, name[j], label[j]  ))
#                     xname = name[j].decode('utf-8')  #图片上显示的xlable名字
#                     for n in self.__get_clound_name(label[j]):
#                         xname += " " + n   
#                     plt.xlabel(xname, fontproperties="SimHei")
#                     plt.imshow(image[j])
#                     plt.show()
        except tf.errors.OutOfRangeError:
            print("end!读完了...")        

        print('图片调用频率：\n', p_freq)
        
        

if __name__ == '__main__':
   
    dg = DatasetGenerator()
    
    dg._test_read_write()
#     t = dg._separete(Pic.get_train_NLs('single'))

#     ts = ['xs','vs','xm','vm','w']
#     for k in ts:
#         t = dg.get_tfr_file_names(k)
#     #     t = dg.x_single_NLs


    
#     #比较resize前后的图片
#     df = dg._test_get_img('xs', 15)
#     before, after = [], []
#     for index, (name, label) in df.iterrows():
#         img = Pic.read_image(name)
#         before.append(img)
#         img = Pic._resize(img)
#         after.append(img)
#     Pic.show_double_img(before, after, 3)

    
    
    
    
