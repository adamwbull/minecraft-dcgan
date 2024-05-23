import zipfile
import os
import sys

def unzip_file(zip_path, extract_to):
    """
    Unzips a ZIP file to the specified directory, creating any necessary directories.
    
    Args:
    zip_path (str): The path to the ZIP file.
    extract_to (str): The directory to extract the files to.
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Extract all the contents into the directory
        for member in zip_ref.infolist():
            # Build the path for extraction
            extracted_path = os.path.join(extract_to, member.filename)
            
            # Create directories if they don't exist
            if member.is_dir():
                os.makedirs(extracted_path, exist_ok=True)
            else:
                # Create the parent directory of the file if it doesn't exist
                parent_dir = os.path.dirname(extracted_path)
                if not os.path.exists(parent_dir):
                    os.makedirs(parent_dir, exist_ok=True)

                # Extract the file
                with open(extracted_path, 'wb') as f:
                    f.write(zip_ref.read(member.filename))

if __name__ == "__main__":
    # Check if the arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python remoteunzip.py <path_to_zip> <target_directory>")
        sys.exit(1)

    zip_path = sys.argv[1]
    target_directory = sys.argv[2]

    # Unzip the file
    unzip_file(zip_path, target_directory)
    print(f"Unzipped '{zip_path}' to '{target_directory}'")
