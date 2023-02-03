from __future__ import print_function
import torch

print(torch.cuda.is_available())
print(torch.__version__)
print(torch.version.cuda)

#创建空矩阵
x = torch.empty(5, 3)
#随机初始化一个矩阵
rand_x = torch.rand(5, 3)
#创建为0矩阵，类型是long
zero_x = torch.zeros(5, 3, dtype=torch.long)
#直接传递tensor数值创建
tensor_x = torch.tensor([[5.5, 3], [3.5, 2]])

#和numpy数组对换
numpy_x = x.numpy()
print(x)
print(numpy_x)
