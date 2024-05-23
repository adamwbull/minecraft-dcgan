import os
import re
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

def chart_log_file(file_path):
    losses_d, losses_g = defaultdict(list), defaultdict(list)
    session_offset = 0  # To handle epoch continuity across sessions
    current_max_epoch = 0  # Track the maximum epoch in the current session

    enable_logging = False
    #print('file_path:',file_path)
    if r"./models\dataset-64-1244\200epochs\ll_mb.log" in file_path:
        enable_logging = True

    with open(file_path, 'r') as file:
        for line in file:
            # Detect start of a new session and adjust epoch offset
            if 'World size: 2' in line:
                if enable_logging:
                    print('')
                    print('!!!!new session offset:',session_offset)
                    print('')
                session_offset = current_max_epoch
                continue
            
            match = False

            if 'GPU' in line:
                match = re.search(r'\[(.*?)\] GPU \d+: Epoch (\d+)/\d+ \| Discriminator Loss: (.*?) \| Generator Loss: (.*)', line)
            #else:
                #match = re.search(r'\[(.*?)\].*?Epoch (\d+)/\d+ \| Discriminator Loss: (.*?) \| Generator Loss: (.*)', line)

            if match:
                _, epoch, loss_d, loss_g = match.groups()
                epoch_adjusted = int(epoch) + session_offset  # Adjust epoch for continuity
                current_max_epoch = max(current_max_epoch, epoch_adjusted)  # Update max epoch for current session
                
                losses_d[epoch_adjusted].append(float(loss_d))
                losses_g[epoch_adjusted].append(float(loss_g))
                
                if enable_logging and (len(losses_d[epoch_adjusted]) > 2 or len(losses_g[epoch_adjusted]) > 2):
                    print('epoch:',epoch)
                    print('session_offset:',session_offset)
                    print('epoch_adjusted:',epoch_adjusted)
                    print('losses_d[epoch_adjusted]:',len(losses_d[epoch_adjusted]))
                    print('losses_g[epoch_adjusted]:',len(losses_g[epoch_adjusted]))

    # Averaging losses across GPUs for each epoch
    epochs = sorted(losses_d.keys())
    avg_losses_d = [sum(losses_d[epoch])/len(losses_d[epoch]) for epoch in epochs]
    avg_losses_g = [sum(losses_g[epoch])/len(losses_g[epoch]) for epoch in epochs]

    if avg_losses_d and avg_losses_g:
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, avg_losses_d, label='Discriminator Loss')
        plt.plot(epochs, avg_losses_g, label='Generator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Over Epochs')
        plt.legend()

        # Compute the new file path
        new_file_path = file_path.replace('.log', '.png').replace('./models', './images')
        os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
        plt.savefig(new_file_path)
        plt.close()

# The process_directory function remains the same
def process_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.log'):
                log_path = os.path.join(root, file)
                chart_log_file(log_path)

if __name__ == "__main__":
    process_directory('./models')
