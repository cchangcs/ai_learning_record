import torch
import numpy as np

# np_data = np.arange(6).reshape((2, 3))
# torch_data = torch.from_numpy(np_data)
# tensor2array = torch_data.numpy()
# print(
#     '\nnumpy', np_data,
#     '\ntorch', torch_data,
#     '\ntensor2array', tensor2array
# )


# data = [-1, -2, 1, 2]
# tensor = torch.FloatTensor(data)
#
# # abs
# print(
#     '\nabs',
#     '\nnumpy', np.abs(data),
#     '\ntorch', torch.abs(tensor)
# )
#
# # sin
# print(
#     '\nsin',
#     '\nnumpy', np.sin(data),
#     '\ntorch', torch.sin(tensor)
# )
#
# # mean
# print(
#     '\nmean',
#     '\nnumpy', np.mean(data),
#     '\ntorch', torch.mean(tensor)
# )

data = [[1, 2], [3, 4]]
tensor = torch.FloatTensor(data)

# multiply
print(
    '\nmultiply',
    '\nnumpy', np.matmul(data, data),
    '\ntorch', torch.mm(tensor, tensor)
)
data2 = np.array(data)
# dot
print(
    '\ndot',
    '\nnumpy', data2.dot(data2),
    '\ntorch', tensor.dot(tensor)
)
