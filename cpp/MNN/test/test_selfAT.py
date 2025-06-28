import sys
# pwd 
import os
current_working_directory = os.getcwd()
print("Current working directory:", current_working_directory)

sys.path.append("../python/")  # Adjust the path as necessary
from MNN import MSelfAT
import numpy as np

# embed size 768
at = MSelfAT()

# set seed for reproducibility
np.random.seed(42)

# B S E
input = np.random.uniform(-1, 1, (1, 5, 768)).astype(np.float32)
output = at.forward(input)[0]

print("Input shape:", input.shape, input.dtype)
print("Output shape:", output.shape, output.dtype)
print("Input strides:", input.strides)
print("Output stride:", output.strides)

save_path = "build/test/temp"
print("at.k_proj.weight.type:", at.k_proj.weight.dtype)
at.k_proj.weight.tofile(save_path + "/k_proj_weight.bin")
at.v_proj.weight.tofile(save_path + "/v_proj_weight.bin")
at.q_proj.weight.tofile(save_path + "/q_proj_weight.bin")
at.out_proj.weight.tofile(save_path + "/out_proj_weight.bin")
at.out_proj.bias.tofile(save_path + "/out_proj_bias.bin")
input.tofile(save_path + "/input.bin")
output.tofile(save_path + "/output.bin")
