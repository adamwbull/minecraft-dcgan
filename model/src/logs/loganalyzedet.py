import os
import re
from statistics import mean, median, mode
from datetime import datetime, timedelta

def parse_log_file(file_path):
    losses_d, losses_g, epochs = [], [], set()
    best_losses = {'best_loss_g': [], 'best_loss_d': [], 'best_loss_d_closest_to_half': []}
    window_start, window_end, total_training_duration = None, None, timedelta()
    max_epoch_seen = -1
    total_epochs_count = 0

    with open(file_path, 'r') as file:
        epoch = 0
        loss_d = 0
        loss_g = 0
        
        for line in file:
            # Match training log entries
            match = re.search(r'\[(.*?)\].*?Epoch (\d+)/\d+ \| Discriminator Loss: (.*?) \| Generator Loss: (.*)', line)
            if match:
                
                # Standard log entry processing
                timestamp_str, epoch, loss_d, loss_g = match.groups()
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')

                if not window_start:
                    window_start = timestamp
                elif timestamp - window_end > timedelta(hours=1):
                    total_training_duration += window_end - window_start
                    window_start = timestamp

                epoch = int(epoch)
                if epoch < max_epoch_seen:  # Epoch reset detected
                    total_epochs_count += len(epochs)
                    epochs.clear()
                max_epoch_seen = max(max_epoch_seen, epoch)

                loss_d, loss_g = float(loss_d), float(loss_g)
                losses_d.append(loss_d)
                losses_g.append(loss_g)
                epochs.add(epoch)
                window_end = timestamp

            # Match best loss save entries
            for key in best_losses.keys():
                if f'Saved new {key} checkpoint' in line:
                    best_losses[key].append((epoch, loss_d, loss_g))

        if window_start and window_end:
            total_training_duration += window_end - window_start

    total_epochs_count += len(epochs)  # Add epochs from the last series

    report = {
        'mean_loss_d': mean(losses_d),
        'median_loss_d': median(losses_d),
        'mode_loss_d': mode(losses_d),
        'mean_loss_g': mean(losses_g),
        'median_loss_g': median(losses_g),
        'mode_loss_g': mode(losses_g),
        'total_epochs': total_epochs_count,
        'total_time': total_training_duration
    }

    # Include stats for the last saved state of each best loss file
    for key, values in best_losses.items():
        if values:
            last_saved = values[-1]
            report[f'{key}_epoch'] = last_saved[0]
            report[f'{key}_loss_d'] = last_saved[1]
            report[f'{key}_loss_g'] = last_saved[2]

    return report


def format_report(report):
    formatted_report = []

    if 'untrained' in report:
        return "N/A"

    formatted_report.append(f"Mean Discriminator Loss: {report['mean_loss_d']:.4f}")
    formatted_report.append(f"Mean Generator Loss: {report['mean_loss_g']:.4f}")
    formatted_report.append(f"Total Epochs: {report['total_epochs']}")
    formatted_report.append(f"Total Training Time: {report['total_time']}")

    # Add information for best_loss_g, best_loss_d, and best_loss_closest_to_half
    for key in ['best_loss_g', 'best_loss_d', 'best_loss_d_closest_to_half']:
        if f'{key}_epoch' in report:
            formatted_report.append(f"\n### {key.upper()} Last Saved Stats:")
            formatted_report.append(f"Epoch: {report[f'{key}_epoch']}")
            formatted_report.append(f"Discriminator Loss: {report[f'{key}_loss_d']:.4f}")
            formatted_report.append(f"Generator Loss: {report[f'{key}_loss_g']:.4f}")

    return '\n'.join(formatted_report)

def analyze_logs(models_folder):
    reports = {}
    for filename in os.listdir(models_folder):
        if filename.endswith('.log'):
            model_name = filename[:-4]
            file_path = os.path.join(models_folder, filename)
            report = parse_log_file(file_path)
            reports[model_name] = report
    return reports

models_folder = './models'
reports = analyze_logs(models_folder)
for model, report in reports.items():
    print(f"### Overall Training Report for {model}:\n{format_report(report)}\n")


