import os
import zipfile
import glob

def zip_files():
    # Define the directories to zip
    log_files = './src/logs/*.log'
    model_files = './src/models/*terminated*.pth.tar'
    
    # Create a new zip file
    with zipfile.ZipFile('download.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add log files
        for file in glob.glob(log_files):
            # Include 'src/logs' in the path inside the ZIP
            zipf.write(file, os.path.relpath(file, '.'))

        # Add model files
        for file in glob.glob(model_files):
            # Include 'src/models' in the path inside the ZIP
            zipf.write(file, os.path.relpath(file, '.'))

# Run the function
zip_files()
