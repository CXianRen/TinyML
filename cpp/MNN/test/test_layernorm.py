import sys
# pwd 
import os
current_working_directory = os.getcwd()
print("Current working directory:", current_working_directory)

sys.path.append("../python/")  # Adjust the path as necessary
from MNN import MLayerNorm
import numpy as np


ln = MLayerNorm(768)

input = np.random.uniform(-1, 1, (1, 5, 768)).astype(np.float32)

output = ln.forward(input)

print(ln.gamma.shape, ln.beta.shape)
print(input.shape, output.shape)

save_path = "build/test/temp"
ln.gamma.tofile(os.path.join(save_path, "gamma.bin"))
ln.beta.tofile(os.path.join(save_path, "beta.bin"))
input.tofile(os.path.join(save_path, "input.bin"))
output.tofile(os.path.join(save_path, "output.bin"))