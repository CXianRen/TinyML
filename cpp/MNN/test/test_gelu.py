import sys
# pwd 
import os
current_working_directory = os.getcwd()
print("Current working directory:", current_working_directory)

sys.path.append("../python/")  # Adjust the path as necessary
from MNN import GELUActivation
import numpy as np


model = GELUActivation()

input = np.random.uniform(-1, 1, (1, 5, 768)).astype(np.float32)

output = model.forward(input)
print("Input shape:", input.shape)
print("Output shape:", output.shape)
save_path = "build/test/temp"
input.tofile(os.path.join(save_path, "input.bin"))
output.tofile(os.path.join(save_path, "output.bin"))