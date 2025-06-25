import numpy as np

weight = np.random.rand(1000, 768).astype(np.float32)

input_ids = np.array([[1, 3 , 5], [2, 4, 6]], dtype=np.int32)

output = weight[input_ids]

print("Input shape:", input_ids.shape)
print("Output shape:", output.shape)
print("weight shape:", weight.shape)

# Save the data to .
save_path = "build/test/temp"
weight.tofile(f"{save_path}/weight.bin")
input_ids.tofile(f"{save_path}/input.bin")
output.tofile(f"{save_path}/output.bin")