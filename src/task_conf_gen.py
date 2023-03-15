import os
import json

# 定义一个空字典，用来存储分类和文件的对应关系
result = {}
folder = '/workspace/IE_data_v2'


# 遍历第一层目录，获取每个分类的文件夹名
for category in os.listdir(folder):
    # 拼接分类的文件夹路径
    category_path = os.path.join(folder, category)
    # 判断是否是文件夹，如果是，则继续遍历
    if os.path.isdir(category_path):
        # 定义一个空列表，用来存储该分类下的所有文件名
        files = []
        # 遍历第二层目录，获取每个文件的文件名
        for file in os.listdir(category_path):
            # 拼接文件的路径
            file_path = os.path.join(category_path, file)
            # 判断是否是文件，如果是，则添加到列表中
            if os.path.isdir(file_path):
                files.append({"sampling strategy": "random", "dataset name": file})
        # 把分类和文件列表作为键值对添加到字典中
        result[category] = files

result_str = json.dumps(result)
print(result_str)
