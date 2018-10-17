import torch
from torch.autograd import Variable

tensor = torch.FloatTensor([[1, 2], [3, 4]])
# requires_grad表示需不需要将定义的variable放到反向传播中
variable = Variable(tensor, requires_grad=True)

# mean
t_out = torch.mean(tensor * tensor)
v_out = torch.mean(variable * variable)

print(t_out)
print(v_out)

v_out.backward()
# v_out = 1/4 * sum(var * var)
# d(v_out)/d(var) = 1/4 * 2 * variable = variable /2
print(variable.grad)

print(variable.data)

# 将variable转换为numpy不能像Tensor一样使用 tensor。numpy
print(variable.data.numpy())


