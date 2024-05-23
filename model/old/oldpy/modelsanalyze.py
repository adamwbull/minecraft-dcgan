import os
import torch

def parse_checkpoint(file_path):
    checkpoint = torch.load(file_path)
    return {
        'best_loss_d_closest_to_half': checkpoint.get('best_loss_d_closest_to_half', 'N/A'),
        'best_loss_g': checkpoint.get('best_loss_g', 'N/A'),
        'best_loss_d': checkpoint.get('best_loss_d', 'N/A'),
        'loss_d': checkpoint.get('loss_d', 'N/A'),
        'loss_g': checkpoint.get('loss_g', 'N/A'),
        'epoch': checkpoint.get('epoch', 'N/A')
    }

def format_report(report):
    formatted_report = []
    for key, value in report.items():
        formatted_report.append(f"{key.replace('_', ' ').title()}: {value}")
    return '\n'.join(formatted_report)

def generate_model_reports(models_folder):
    reports = {}
    for filename in os.listdir(models_folder):
        if filename.endswith('.pth.tar'):
            model_name = filename[:-len('.pth.tar')]
            file_path = os.path.join(models_folder, filename)
            report = parse_checkpoint(file_path)
            reports[model_name] = report
    return reports

def main():
    models_folder = './models'
    reports = generate_model_reports(models_folder)
    for model, report in reports.items():
        print(f"### Stats for model state {model}:\n{format_report(report)}\n")

if __name__ == "__main__":
    main()
