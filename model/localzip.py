import os
import zipfile
import sys

# For zipping src and logs on local, excluding certain directories.

def zip_directory(path, ziph):
    base_path = os.path.normpath(path)
    for root, dirs, files in os.walk(base_path):
        # Skip the specified directories
        dirs[:] = [d for d in dirs if os.path.join(root, d) not in [os.path.join(base_path, 'models'), os.path.join(base_path, 'generations'), os.path.join(base_path, 'logs')]]
        
        for file in files:
            # Create a relative path for files to maintain the directory structure
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, base_path)
            ziph.write(file_path, path+'/'+relative_path)

def main(zip_name):
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add src directory and its contents, excluding specified subdirectories
        zip_directory('./src', zipf)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python zip.py <zip_name>")
        sys.exit(1)

    # Ensures that the zip file is saved in the top-level directory
    zip_name = os.path.join('.', sys.argv[1])
    main(zip_name)
