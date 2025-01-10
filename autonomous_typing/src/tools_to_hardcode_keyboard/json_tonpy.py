import json
import numpy as np

# Load the JSON file
with open("keyboard_layout.json", "r") as json_file:
    data = json.load(json_file)

# Convert JSON data to a NumPy array
# Assuming you want to preserve key-value pairs as a dictionary in .npy
data_array = {key: np.array(value) for key, value in data.items()}
a=data_array.values()
b=data_array.keys()
print(a)
print(type(data_array[b[0]]))
# Save as .npy file

print("JSON data has been successfully converted to output.npy")
