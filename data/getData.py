# import openxlab
# 从openxlab下载数据集
# openxlab.login(ak="m1qxgyeqpqxmwb8y2wmx",
# sk="aoklozl6bpzdxjbqbd6dy9mlxrywa9vdenrv85yq")
from openxlab.dataset import get

get(dataset_repo='OpenDataLab/MNIST-M', target_path='../checkpoint')
# target_path：下载文件指定的本地路径
