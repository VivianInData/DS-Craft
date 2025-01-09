import gdown
import os

# Google Drive file ID and output file
file_id = "1A2B3C4D5E6F7G8H9"
output_file = "X_item_features.npy"

# Construct the URL
url = f"https://drive.google.com/uc?id={file_id}"

# Check if the file exists
if not os.path.exists(output_file):
    print(f"{output_file} not found. Downloading from Google Drive...")
    gdown.download(url, output_file, quiet=False)
else:
    print(f"{output_file} already exists.")
