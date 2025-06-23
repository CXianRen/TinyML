import numpy as np

# input = np.array([
#     [1, 2, 3],
#     [4, 5, 6],
#     [7, 8, 9]
# ])

# # 数值稳定性：减去每行的最大值
# shifted = input - np.max(input, axis=1, keepdims=True)

# exp = np.exp(shifted)
# softmax = exp / np.sum(exp, axis=1, keepdims=True)

# print(softmax)

# [S, HN, HS] [2, 2, 3]
# input = np.array([
#     [
#         [1, 2, 3],
#         [4, 5, 6]
#     ],
#     [
#         [7, 8, 9],
#         [10, 11, 12]
#     ],
# ])

# print("input:")
# print("input stride:", input.strides)
# print(input)
# # transpose the input to [HN, S, HS]

# print("Transposed input:")
# print("Transposed input stride:", input.transpose((1, 0, 2)).strides)
# input_transposed = np.transpose(input, (1, 0, 2))
# print(input_transposed)

# import numpy as np
# # (3, 2).T -> (2, 2, 2, 3)
# a = np.array([
#               [1, 2], 
#               [3, 4], 
#               [5, 6]])

# print("a shape:", a.shape)
# print("a strides:", a.strides)
# a = a.T # Transpose to (3, 2)
# print("Transposed a shape:", a.shape)
# print("Transposed a strides:", a.strides)

# broadcasted = np.broadcast_to(a, (2, 2, 2, 3))

# # print("Broadcasted array:\n", broadcasted)
# # Broadcasted shape: (2, 3)
# print("Broadcasted shape:", broadcasted.shape)
# print("Broadcasted strides:", broadcasted.strides)
# Broadcasted strides: (0, 0)

# import numpy as np

# # [2, 2, 3] 
# a = np.array([[[1, 2, 3],
#                [4, 5, 6]], 
#               [[1, 2, 3],
#                [4, 5, 6]]])
    
# b = np.ones((1, 2, 2, 3))

# r = np.matmul(b, np.transpose(a, (0, 2, 1)))
# print("a shape:", a.shape)
# print("b shape:", b.shape)
# print("Result shape:", r.shape)
# print("Result strides:", r.strides)
# print("Result:\n", r)


# import numpy as np

# # [2, 2, 3] 
# a = np.random.rand(2, 1, 3, 4)
# print("a shape:", a.shape)
# print("a strides:", a.strides)   
# print("a data address:", a.__array_interface__['data'][0]) 
# b = a[1][0]
# print("b shape:", b.shape)
# print("b strides:", b.strides)
# print("b data address:", b.__array_interface__['data'][0])

import numpy as np

a = np.array([
              # [[1.0, 2.0, 3.0],
              #  [4.0, 5.0, 6.0]],

              [[7.0, 8.0, 9.0],
               [10.0, 11.0, 12.0]]])  # shape: (2, 2, 3)

b = np.array([
              # [[1.0, 4.0],
              #  [2.0, 5.0],
              #  [3.0, 6.0]],

              [[7.0, 10.0],
               [8.0, 11.0],
               [9.0, 12.0]]])  # shape: (2, 3, 2)

c = np.matmul(a, b)  # shape: (2, 2, 2)

print("Output shape:", c.shape)
print("Result:\n", c)