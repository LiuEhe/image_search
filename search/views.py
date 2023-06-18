# 导入处理base64编码的模块，主要用于图片编码
import base64

# 导入处理字节流的模块，主要用于处理图片数据
import io

# 导入numpy模块，主要用于处理数组和矩阵计算
import numpy as np

# 导入Django的功能，主要用于响应http请求和处理文件存储
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage

# 导入faiss模块，主要用于快速查找最近邻
import faiss

# 导入tensorflow的图像处理模块
from keras.preprocessing import image
from keras.utils import load_img

# 引入 utils 中的模块与函数
from .utils import process_image, get_img_files, get_faiss_index, get_model

# 加载图像文件列表
image_files = get_img_files()

# 加载Faiss索引
faiss_index = get_faiss_index()

# 加载vgg16模型
vgg16 = get_model()

# 定义主函数，处理用户上传图片的请求，并返回相似图片
def index(request):
    # 判断请求方式是否为POST
    if request.method == 'POST':
        # 获取上传的图片
        uploaded_image = request.FILES['query_image']
        
        # 创建文件存储对象，从settings.MEDIA_ROOT获取media文件夹的位置，并把上传图像存入
        fs = FileSystemStorage(location=settings.MEDIA_ROOT)
        filename = fs.save(uploaded_image.name, uploaded_image)

        # 获取存储的上传文件的位置
        uploaded_image_url = fs.path(filename)

        # 提取图片 uploaded_image_url 的特征, 并把特征放到一个空数组中
        feature = process_image(uploaded_image_url, vgg16)
        X = np.empty((1,512), dtype=np.float32)
        X[0] = feature

        # 使用Faiss索引进行搜索，返回最相似的11张图片
        # 对X进行标准化X，余弦相似性必备步骤
        faiss.normalize_L2(X)
        D, I = faiss_index.search(X, 11)

        similar_images = []
        # 对于搜索结果中的每一张相似图片和相应的相似度
        for index, similarity in zip(I[0][1:], D[0][1:]):
            # 使用索引从图像文件列表中加载相似图片，并调整图片大小为224x224
            similar_img =load_img(image_files[index], target_size=(224, 224))
            # 创建一个内存文件对象
            buffered = io.BytesIO()
            # 将相似图片保存为JPEG格式的内存文件对象
            similar_img.save(buffered, format="JPEG")
            # 获取内存文件对象的二进制数据，并转换为base64编码，然后解码为字符串
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            # 将图片的base64编码和相似度添加到相似图片列表中，相似度乘以100转换为百分比
            similar_images.append((img_base64, similarity*100))


        # 将上传的图片转为base64编码
        with open(uploaded_image_url, "rb") as f:
            uploaded_image_base64 = base64.b64encode(f.read()).decode()

        # 返回渲染后的页面，携带上传的图片和相似图片的数据
        return render(request, 'search/index.html', {
            'uploaded_image': uploaded_image_base64,
            'images': similar_images,
        })

    # 如果请求方式不是POST，返回渲染后的页面
    return render(request, 'search/index.html')