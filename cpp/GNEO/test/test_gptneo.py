import sys
# pwd 
import os
current_working_directory = os.getcwd()
print("Current working directory:", current_working_directory)

sys.path.append("../python/")  # Adjust the path as necessary
from GPTNeo import GPTNeoModel
import numpy as np


model = GPTNeoModel(4, 768)
model.load("../model_state_dict")

input_ids = np.array([[ 7454, 2402,257,640,11]]).astype(np.int32)  # Example input
position_ids = np.arange(input_ids.shape[1]).reshape(1, -1).astype(np.int32)  # Position IDs

print("Input shape:", input_ids.shape)
print("Position IDs shape:", position_ids.shape)
output = model(input_ids, position_ids=position_ids)
print("Output shape:", output.shape)
print("Output:", output)
next_token_logits = output[:, -1, :]
next_token = np.argmax(next_token_logits, axis=-1).reshape(-1, 1)
print("Next token:", next_token)


# save the output to a file
save_path = "build/test/temp"
output.tofile(os.path.join(save_path, "output.bin"))