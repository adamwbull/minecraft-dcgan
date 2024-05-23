import os
import zipfile
import glob
import argparse

parser = argparse.ArgumentParser(description='zip a log directory in src')
parser.add_argument('dir', type=str, help='name of log folder inside src')
args = parser.parse_args()

def zip_files():
    # Define the directories to zip
    log_files = './src/'+args.dir+'/models/*.log'
    error_files = './src/'+args.dir+'/errors/*.log'
    condor_files = './src/'+args.dir+'/condor/*.log'
    base_files = './src/'+args.dir+'/*.log'
    
    # Create a new zip file
    with zipfile.ZipFile('download'+args.dir+'.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:

        for file in glob.glob(log_files):
            zipf.write(file, os.path.relpath(file, '.'))

        for file in glob.glob(error_files):
            zipf.write(file, os.path.relpath(file, '.'))

        for file in glob.glob(condor_files):
            zipf.write(file, os.path.relpath(file, '.'))

        for file in glob.glob(base_files):
            zipf.write(file, os.path.relpath(file, '.'))

# Run the function
zip_files()
