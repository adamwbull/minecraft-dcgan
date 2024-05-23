import os
import re
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

def chart_log_file(file_path):
    # Using defaultdict to handle multiple GPU entries per epoch
    losses_d, losses_g = defaultdict(list), defaultdict(list)
    timestamps = defaultdict(list)

    with open(file_path, 'r') as file:
        for line in file:
            if 'GPU' in line:
                # Alternative parsing for logs with GPU info
                match = re.search(r'\[(.*?)\] GPU \d+: Epoch (\d+)/\d+ \| Discriminator Loss: (.*?) \| Generator Loss: (.*)', line)
            else:
                match = re.search(r'\[(.*?)\].*?Epoch (\d+)/\d+ \| Discriminator Loss: (.*?) \| Generator Loss: (.*)', line)

            if match:
                timestamp_str, epoch, loss_d, loss_g = match.groups()
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                epoch = int(epoch)
                losses_d[epoch].append(float(loss_d))
                losses_g[epoch].append(float(loss_g))
                timestamps[epoch].append(timestamp)

    # Averaging losses across GPUs for each epoch
    avg_losses_d = [sum(losses_d[epoch])/len(losses_d[epoch]) for epoch in sorted(losses_d)]
    avg_losses_g = [sum(losses_g[epoch])/len(losses_g[epoch]) for epoch in sorted(losses_g)]
    avg_timestamps = [min(timestamps[epoch]) for epoch in sorted(timestamps)]

   
    if avg_losses_d and avg_losses_g:
        fig, ax1 = plt.subplots(figsize=(10, 5))

        color = 'tab:blue'
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Discriminator Loss', color=color)
        lns1 = ax1.plot(avg_timestamps, avg_losses_d, label='Discriminator Loss', color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:orange'
        ax2.set_ylabel('Generator Loss', color=color)  # we already handled the x-label with ax1
        lns2 = ax2.plot(avg_timestamps, avg_losses_g, label='Generator Loss', color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        # Combine the legends from both axes
        lns = lns1 + lns2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc=0)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.title('Loss Over Time')
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
