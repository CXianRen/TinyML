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
input = np.array([
    [
        [1, 2, 3],
        [4, 5, 6]
    ],
    [
        [7, 8, 9],
        [10, 11, 12]
    ],
])

print("input:")
print("input stride:", input.strides)
print(input)
# transpose the input to [HN, S, HS]

print("Transposed input:")
print("Transposed input stride:", input.transpose((1, 0, 2)).strides)
input_transposed = np.transpose(input, (1, 0, 2))
print(input_transposed)