import os

def check_file_path(file_path):
    if os.path.exists(file_path):
        print(f"File exists: {file_path}")
    else:
        print(f"File does not exist: {file_path}")

# Example usage
file_path = r'D:\Project_githup\ma\DL_P4(Fashion MNIST - Image Classification CNN )\templates\index.html'
check_file_path(file_path)
