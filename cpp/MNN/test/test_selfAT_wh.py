import sys
# pwd 
import os
current_working_directory = os.getcwd()
print("Current working directory:", current_working_directory)

sys.path.append("../python/")  # Adjust the path as necessary

import numpy as np

from MNN import MSelfAT
# set seed for reproducibility
np.random.seed(42)

# embed size 768
at = MSelfAT()


# B S E
input = np.random.uniform(-1, 1, (1, 5, 768)).astype(np.float32)
output, k_h, v_h = at.forward(input)

input_single = np.random.uniform(-1, 1, (1, 1, 768)).astype(np.float32)
output_single, k_h_single, v_h_single = at.forward(input_single, k_h, v_h)

print("Input shape:", input.shape, input.dtype)
print("Output shape:", output.shape, output.dtype)
print("Input strides:", input.strides)
print("Output stride:", output.strides)

save_path = "build/test/temp"
at.k_proj.weight.tofile(save_path + "/k_proj_weight.bin")
at.v_proj.weight.tofile(save_path + "/v_proj_weight.bin")
at.q_proj.weight.tofile(save_path + "/q_proj_weight.bin")
at.out_proj.weight.tofile(save_path + "/out_proj_weight.bin")
at.out_proj.bias.tofile(save_path + "/out_proj_bias.bin")

input_single.tofile(save_path + "/input_single.bin")
output_single.tofile(save_path + "/output_single.bin")
k_h.tofile(save_path + "/k_h.bin")
v_h.tofile(save_path + "/v_h.bin")
k_h_single.tofile(save_path + "/k_h_single.bin")
v_h_single.tofile(save_path + "/v_h_single.bin")
