# train.py
import numpy as np
import faiss
from utils import get_model, get_img_files, batch_process_images
from django.conf import settings
import tensorflow as tf
import os
import glob




# 1. 获取 VGG16 模型
model = get_model()

# 2. 获取图像文件列表
img_files=glob.glob('./static/data/data/*.jpg')
img_files=sorted(img_files)



# 3. 批量处理图像并提取特征
features = batch_process_images(img_files, model)

# 4. 创建 Faiss 索引
dimension = features.shape[1]  # 获取特征向量的维度
index = faiss.IndexFlatIP(dimension)  # 创建 余弦 距离的 Faiss 索引

#标准化数据
faiss.normalize_L2(features)

# # 初始化GPU资源，并将索引转移到GPU上
# gpu_res = faiss.StandardGpuResources()
# gpu_index = faiss.index_cpu_to_gpu(gpu_res, device=0, index=index)
# gpu_index.add(features) # 将数据添加到GPU索引中
index.add(features)  # 将特征向量添加到索引中


# 5. 保存 Faiss 索引到文件
#faiss.write_index(index, settings.FAISS_INDEX_FILEPATH)
faiss.write_index(index, './search/faiss_index_database')
#print(f"Faiss 索引文件已保存到 {settings.FAISS_INDEX_FILEPATH}")
#print(index.shape)
print(f"Faiss 索引文件已保存到 ./search/faiss_index_database")

