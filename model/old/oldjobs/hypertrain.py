import subprocess
import argparse
import time
import gc
from variables.hyperparameters import hyperparam_grid

# Function to run train.py with different hyperparameters
def run_training(hyperparams, process_dict, max_duration):
    cmd_train = ['/cluster/research-groups/deneke/minecraft-gan/dcgan_pyenv/bin/python', 'train.py', hyperparams['trial_name'],
                 '--epochs', str(hyperparams['epochs']),
                 '--lr', str(hyperparams['lr']),
                 '--batch_size', str(hyperparams['batch_size'])]

    #cmd_generate = ['/cluster/research-groups/deneke/minecraft-gan/dcgan_pyenv/bin/python', 'generate.py', hyperparams['trial_name']]

    # Start the training process
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
    print(timestamp + " Spawning train.py " + hyperparams['trial_name'] + " --epochs " + str(hyperparams['epochs']) + " --lr " + str(hyperparams['lr']) + " --batch_size " + str(hyperparams['batch_size']))
    train_process = subprocess.Popen(cmd_train)
    process_dict[train_process.pid] = {'process': train_process, 'start_time': time.time(), 'type': 'train', 'cmd': cmd_train, 'hyperparams': hyperparams, 'recursion_level': 0}

    # Monitor training process and start generate process after completion
    monitor_processes(process_dict, max_duration)

    #timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
    #print(timestamp + " Spawning generate.py " + hyperparams['trial_name'])
    #generate_process = subprocess.Popen(cmd_generate)
    #process_dict[generate_process.pid] = {'process': generate_process, 'start_time': time.time(), 'type': 'generate'}

    # Final monitoring and cleanup
    #monitor_processes(process_dict, max_duration)
    cleanup_processes(process_dict)

def monitor_processes(process_dict, max_duration):
    all_processes_completed = False

    while not all_processes_completed:
        all_processes_completed = True
        current_time = time.time()

        for pid, proc_info in list(process_dict.items()):
            if proc_info['process'].poll() is None:  # Process is still running
                all_processes_completed = False
                if current_time - proc_info['start_time'] > max_duration:
                    proc_info['process'].terminate()  # Terminate if exceeds max duration
                    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
                    print(timestamp + " Terminating " + proc_info['type'] + " process.")
                    del process_dict[pid]
            else:
                # Check if process completed early
                elapsed_time = current_time - proc_info['start_time']
                if elapsed_time < max_duration and proc_info['type'] == 'train':
                    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
                    print(timestamp + " Restarting training due to early completion: " + str(elapsed_time) + "/" + str(max_duration))
                    restart_training(proc_info['hyperparams'], process_dict, max_duration, proc_info.get('recursion_level', 0) + 1)
                del process_dict[pid]

        time.sleep(1)  # Sleep for a short duration to prevent busy waiting

def restart_training(hyperparams, process_dict, max_duration, recursion_level):

    if recursion_level > 10:
        timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
        print(timestamp + " Recursion level exceeded for " + hyperparams['trial_name'] + ". Moving to next model.")
        return
    
    cmd_train = ['/cluster/research-groups/deneke/minecraft-gan/dcgan_pyenv/bin/python', 'train.py', hyperparams['trial_name'],
                 '--epochs', str(hyperparams['epochs']),
                 '--lr', str(hyperparams['lr']),
                 '--batch_size', str(hyperparams['batch_size'])]
    
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
    print(timestamp + " Spawning train.py " + hyperparams['trial_name'] + " --epochs " + str(hyperparams['epochs']) + " --lr " + str(hyperparams['lr']) + " --batch_size " + str(hyperparams['batch_size']))
    train_process = subprocess.Popen(cmd_train)
    process_dict[train_process.pid] = {'process': train_process, 'start_time': time.time(), 'type': 'train', 'cmd': cmd_train, 'hyperparams': hyperparams, 'recursion_level': recursion_level}

    # Continue monitoring
    monitor_processes(process_dict, max_duration)

def cleanup_processes(process_dict):
    # Terminate any remaining processes
    for proc_info in process_dict.values():
        process = proc_info['process']
        process.terminate()
        process.wait()  # Wait for the process to terminate

    process_dict.clear()
    gc.collect()  # Trigger garbage collection

# Argument parsing
parser = argparse.ArgumentParser(description='Hyperparameter tuning script.')
parser.add_argument('-hours', '--hours', type=float, help='Number of hours to run each model for', required=True)
parser.add_argument('-loops', '--loops', type=int, help='Number of times to loop through the grid', required=True)
args = parser.parse_args()

# Main loop for hyperparameter tuning
process_dict = {}
max_duration = args.hours * 3600
for _ in range(args.loops):
    for hyperparams in hyperparam_grid:
        run_training(hyperparams, process_dict, max_duration)

# Final cleanup after all loops
cleanup_processes(process_dict)
