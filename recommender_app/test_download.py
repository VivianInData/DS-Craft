import gdown
import os

# Updated file ID
file_id = "1AIRzNPxLJDmZ7ELSV88soSHTE-IM6QBQ"
output_file = "X_item_features.npy"

url = f"https://drive.google.com/uc?id={file_id}"

if not os.path.exists(output_file):
    print(f"{output_file} not found. Downloading from Google Drive...")
    try:
        gdown.download(url, output_file, quiet=False)
        print(f"Downloaded {output_file} successfully.")
    except Exception as e:
        print(f"Error: {e}")
else:
    print(f"{output_file} already exists.")

