import os
import subprocess
import gzip
import shutil
import datetime
from argparse import ArgumentParser
import subprocess

def gzip_file(src_path, dst_path):
    with open(src_path, 'rb') as f_in:
        with gzip.open(dst_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

def append_timestamp(filename):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    name, ext = os.path.splitext(filename)
    return f"{name}_{timestamp}{ext}"

def transfer_file(local_path, remote_path, direction='to_remote'):
    """
    Transfers a file or directory between the local system and a remote system using scp.

    Args:
    local_path (str): The path to the file or directory on the local system.
    remote_path (str): The path where the file or directory should be transferred on the remote system.
    direction (str, optional): The direction of the transfer. 'to_remote' for local to remote, and vice versa. Defaults to 'to_remote'.
    """
    
    # Determine if the local path is a directory
    is_directory = os.path.isdir(local_path)
    
    # Construct the scp command based on the transfer direction and whether the local path is a directory
    scp_command = ''
    if direction == 'to_remote':
        scp_command = 'scp'
        if is_directory:
            scp_command += ' -r'  # Add the recursive flag for directories
        scp_command += f' {local_path} {remote_path}'
    else:  # direction is from remote to local
        scp_command = f'scp {remote_path} {local_path}'
    
    # Execute the scp command
    subprocess.run(scp_command, shell=True)

def send_dataset():
    local_path = './dataset/schematics.hdf5'
    compressed_path = './dataset/schematics.hdf5.gz'
    remote_path = 'cluster:/cluster/research-groups/deneke/minecraft-gan/dataset/schematics.hdf5.gz'
    gzip_file(local_path, compressed_path)
    transfer_file(compressed_path, remote_path)
    os.remove(compressed_path)

def send_source_code():
    for root, dirs, files in os.walk('./src'):
        for file in files:
            if file.endswith('.py') or file.endswith('.job'):
                local_path = os.path.join(root, file)
                remote_dir = 'cluster:/cluster/research-groups/deneke/minecraft-gan/src/' + '/'.join(root.split('/')[1:])
                transfer_file(local_path, remote_dir)

def download_models_logs(list_script_path, host_file, identity_file, remote_host, local_model_dir, local_log_dir):
    # Command to list model and log files
    execute_script_command  = f"bash {list_script_path}"
    
    # Execute SSH command
    
    ssh_command = f"ssh -F {host_file} -i {identity_file} {remote_host} '{execute_script_command}'"
    result = subprocess.run(ssh_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print('stdout:', result.stdout)
    print('stderr:', result.stderr)
    files = result.stdout.splitlines()
    
    # Separate model and log files
    model_files = [f for f in files if f.endswith('.pth.tar')]
    log_files = [f for f in files if f.endswith('.log')]

    # Transfer model files
    for file in model_files:
        local_file_path = os.path.join(local_model_dir, os.path.basename(file))
        if os.path.exists(local_file_path):
            local_file_path = append_timestamp(local_file_path)
        transfer_file(file, local_file_path, direction='to_local')

    # Transfer log files
    for file in log_files:
        local_file_path = os.path.join(local_log_dir, os.path.basename(file))
        if os.path.exists(local_file_path):
            local_file_path = append_timestamp(local_file_path)
        transfer_file(file, local_file_path, direction='to_local')

# Make sure these are all correct.
host_file = '/Users/A/.ssh/config'
identity_file = '/Users/A/.ssh/id_rsa_cluster'
remote_host = 'cluster'
local_model_dir = './src/models'
local_log_dir = './src/logs'
list_script_path = '/cluster/research-groups/deneke/minecraft-gan/list.sh'

parser = ArgumentParser(description='Sync files between local and cluster.')
parser.add_argument('--send-dataset', action='store_true', help='Send the latest dataset to the cluster.')
parser.add_argument('--send-source', action='store_true', help='Send source code and job files to the cluster.')
parser.add_argument('--download-models-logs', action='store_true', help='Download models and logs from the cluster.')
args = parser.parse_args()

if __name__ == "__main__":
    if args.send_dataset:
        send_dataset()
    if args.send_source:
        send_source_code()
    if args.download_models_logs:
        download_models_logs(list_script_path, host_file, identity_file, remote_host, local_model_dir, local_log_dir)
