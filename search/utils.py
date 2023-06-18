# utils.py
# 导入Tensorflow的预处理模块和预训练模型，主要用于处理图片和特征提取
from keras.preprocessing import image
from keras.utils import load_img,img_to_array
from keras.applications.vgg16 import VGG16, preprocess_input
import numpy as np  # NumPy库，用于进行科学计算
import glob  # glob库，用于寻找符合特定规则的文件路径名
import faiss  # Faiss库，用于进行高效的相似度搜索和聚类
from django.conf import settings  # Django配置模块，用于获取Django的设置信息
from tqdm import tqdm  # 导入进度条库tqdm，用于在循环中显示进度条
import time  # 用于计算处理时间
import os  # 用于获取进程 ID
import psutil  # 用于获取内存使用信息
from functools import wraps  # 用于创建装饰器

# 全局变量，用于存储资源
# 用 "_" 强调其为私有变量，不应该直接访问，而应该通过以下函数获取
_model = None # 用来提取特征的模型
_image_files = None # 所有训练图像文件名列表
_faiss_index = None # faiss索引文件对象

# 获取VGG16模型
def get_model(type='vgg16'):
    global _model  # 使用 global 关键字引用全局变量
    if _model is None:  # 如果全局变量未初始化
        if type=='vgg16':
            print("载入vgg16模型")
            # 加载预训练的VGG16模型，不包括顶部的全连接层（include_top=False），因为我们的目标是提取特征，而不是进行分类
            # weights='imagenet' 表示使用在 ImageNet 数据集上预训练的权重，这些权重可以帮助我们更好地提取特征
            # pooling="max" 表示使用最大池化来池化特征图，这可以帮助我们更好地保留特征信息，并且对结果进行大幅度降维
            _model = VGG16(weights='imagenet', include_top=False, pooling="max")
    return _model  # 返回全局变量

# 获取图像文件列表
def get_img_files():
    global _image_files  # 使用 global 关键字引用全局变量
    if _image_files is None:  # 如果全局变量未初始化
        _image_files = glob.glob(settings.TRAINING_IMAGES) # 从settings获取所有训练图像的路径
        _image_files.sort()  # 使用 sort 是为了确保在不同的环境下运行代码的结果一致
    return _image_files  # 返回全局变量

# 使用 faiss.read_index(filepath) 获取FAISS索引文件，赋值给 _faiss_index
def get_faiss_index():
    global _faiss_index  # 使用 global 关键字引用全局变量
    if _faiss_index is None:  # 如果全局变量未初始化
        _faiss_index = faiss.read_index(settings.FAISS_INDEX)  # 从settings获取FAISS索引文件
    return _faiss_index  # 返回全局变量

# 用模型 model 提取 img_file 位置图像文件的特征
def process_image(img_file,model):
    img = load_img(img_file, target_size=(224, 224))  # 加载图像并调整大小为 224x224
    img = img_to_array(img)  # 将图像转换为数组
    img = np.expand_dims(img, axis=0)  # 将数组转换为批次（batch）数据
    img = preprocess_input(img)  # 预处理图像，使其与VGG16模型的输入格式相同
    feature = model.predict(img)  # 使用VGG16模型提取特征
    ##feature = feature.flatten()  # 将特征向量展平
    ##feature = feature / np.linalg.norm(feature)  # 使用L2范数对特征向量进行归一化##


    return feature


# 定义一个装饰器，用于打印batch_process_images运行时的一些重要的信息
def process_info_decorator(func):
    @wraps(func)  # 使用 wraps 保留原始函数的元数据（如名称和文档字符串）
    def wrapper(*args, **kwargs):  # 定义包装函数，它将替换原始函数
        start_time = time.time()  # 记录开始时间
        process = psutil.Process(os.getpid())  # 获取当前进程
        mem_before = process.memory_info().rss / 1024 / 1024  # 记录内存使用情况（以 MB 为单位）

        print("开始批处理图像...")
        result = func(*args, **kwargs)  # 调用原始函数并传递参数

        end_time = time.time()  # 记录结束时间
        mem_after = process.memory_info().rss / 1024 / 1024  # 记录结束时的内存使用情况（以 MB 为单位）
        mem_diff = mem_after - mem_before  # 计算内存使用变化

        num_images = len(args[0])  # 获取处理的图像文件数量
        elapsed_time = end_time - start_time  # 计算耗时

        # 输出统计信息
        print(f"处理完成。共处理 {num_images} 张图像。")
        print(f"耗时：{elapsed_time:.2f} 秒")
        print(f"内存使用变化：{mem_diff:.2f} MB")

        return result  # 返回原始函数的结果

    return wrapper  # 返回包装函数

@process_info_decorator
def batch_process_images(img_files, model, batch_size=32):
    # 批量处理图像
    features = []  # 用于存储提取出的特征
    for i in tqdm(range(0, len(img_files), batch_size)):  # 使用tqdm显示进度
        batch_files = img_files[i:i+batch_size]  # 获取当前批次的文件
        batch_x = np.array([img_to_array(load_img(img_file, target_size=(224, 224))) for img_file in batch_files])  # 加载图像并转换为数组
        batch_x = preprocess_input(batch_x)  # 预处理图像
        batch_features = model.predict(batch_x, verbose=0)  # 提取特征
        features.extend(batch_features)  # 添加到特征列表
    return np.array(features)  # 返回特征数组
