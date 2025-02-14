import os
import pickle
import pprint
import zipfile

# zip_path = "path/to/your_action_name.zip"      # Replace with your zip file path
extract_path = "data/KitchenEnv_seed_0/2025-02-12_20:20:56/traces"       # Destination directory

# Create destination folder if it doesn't exist
# os.makedirs(extract_path, exist_ok=True)

# with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#     zip_ref.extractall(extract_path)

# print(f"Extracted files: {os.listdir(extract_path)}")


pkl_path = os.path.join(extract_path, "data.pkl")  # Adjust the filename if necessary

# Open the pickle file in binary mode and load its content
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

# Check the type of the loaded data (could be a dict, list, etc.)
print("Type of loaded data:", type(data))

# If the data is a dictionary, inspect its keys:
if isinstance(data, dict):
    print("Keys:", data.keys())

# Pretty-print the data for a more readable inspection
# pprint.pprint(data)

print("Length of outer list:", len(data))
if isinstance(data[0], list):
    samples = data[0]
else:
    samples = data
print("Number of elements in samples:", len(samples))

for i, item in enumerate(samples):
    print(f"Element {i}: shape = {item.shape}, dtype = {item.dtype}")

# for i in range(0, len(samples), 2):
#     action = samples[i]         # The array with 4 float values
#     image_flat = samples[i+1]     # The uint8 array likely representing an image
#     print(f"Sample {i//2}: action = {action}")
#     print(f"Sample {i//2}: image shape = {image_flat.shape}")
#     # If you know the intended image dimensions, for example 28x28:
#     try:
#         image = image_flat.reshape(28, 28)
#         # Now, display the image using matplotlib:
#         import matplotlib.pyplot as plt
#         plt.imshow(image, cmap='gray')
#         plt.title(f"Sample {i//2} Image")
#         plt.show()
#     except Exception as e:
#         print("Could not reshape image:", e)
