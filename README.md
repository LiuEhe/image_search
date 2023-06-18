## 项目描述
一个基于Django框架实现的图像相似性搜索网页应用。用户可以通过上传图片到网站，然后该项目会基于预训练的 VGG16 模型提取图像特征，并利用已有图库中的图像特征与上传图片的特征进行比较，计算相似度并呈现给用户。

## 项目运行效果截图
<img src="https://github.com/LiuEhe/image_search/blob/main/result/img_search.jpg" width="384" height="198"><img src="https://github.com/LiuEhe/image_search/blob/main/result/img_search2.jpg" width="384" height="198">

## 功能
- 用户可以通过上传图片查询相似图片
- 实现根据预训练的VGG16模型提取图像特征
- 利用Faiss库创建索引来加速相似图像查询过程
- 支持批量处理图像，创建特征向量索引用于快速检索相似的图片

## 依赖
- Django
- Tensorflow
- Faiss
- glob
- numpy
- psutil

## 使用
1. 克隆本项目到您的本地环境
2. 安装依赖库
3. `static`目录下的data文件夹现在是zip文档，clone项目后先解压该文档，确保解压后路径为:`static/data/data`和`static/data/test`
4. 完成`utils.py`, `train.py`和`views.py`中的`#TODO`代码
5. 运行`train.py`生成`faiss索引文件`faiss_index_database`并把其拷贝到`static/model`文件夹下
6. 通过运行 `python manage.py runserver` 启动Django项目
7. 在浏览器中访问 `http://localhost:8000` 查看网页应用

## 注意
- `static`目录下的data文件夹现在是zip文档，clone项目后先解压该文档，确保解压后路径为:`static/data/data`和`static/data/test`
- 确保项目是基于Python的脚本文件
- 提前准备好图像数据集，并使用预训练的VGG16模型进行特征提取
- 使用Faiss库创建索引以加速相似图像查询过程
- 在使用Django框架时，遵从其目录结构和规范
- 数据集请前往[notion](https://liuehe.notion.site/1a7fee02d1f04d09803fc15d20e49cda?pvs=4)下载
