import numpy as np


weight = np.random.rand(768, 768).astype(np.float32)
bias = np.random.rand(768).astype(np.float32)

input_data = np.random.rand(2, 5, 768).astype(np.float32)

output_data = np.dot(input_data, weight.T) + bias
print("Input Data Shape:", input_data.shape)
print("Weight Shape:", weight.shape)
print("Bias Shape:", bias.shape)
print("Output Data Shape:", output_data.shape)

# Save the data to .
# a.tofile(f"{args.save_path}/a.bin")
save_path = "build/test/temp"
weight.tofile(f"{save_path}/weight.bin")
bias.tofile(f"{save_path}/bias.bin")
input_data.tofile(f"{save_path}/input.bin")
output_data.tofile(f"{save_path}/output.bin")