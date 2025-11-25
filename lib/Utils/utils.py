import os
import logging
from datetime import datetime
import json
import torch
import shutil


def setup_logging(run_dir=None):
    """Setup logging to both console and file with timestamp
    
    Parameters:
        run_dir (str): Directory to save log file. If None, uses 'logs/' directory.
    """
    if run_dir is None:
        logs_dir = os.path.join(os.getcwd(), 'logs')
    else:
        logs_dir = run_dir
    
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create log filename with current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f'logs_{timestamp}.txt')
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file


class MetricsLogger:
    """Logger for tracking metrics and saving to file"""
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.metrics = {}
        
    def add_scalar(self, tag, value, step):
        """Add a scalar metric value to be logged"""
        if tag not in self.metrics:
            self.metrics[tag] = []
        
        self.metrics[tag].append({
            'step': step,
            'value': float(value)
        })
        
        # Log to console and file
        logger = logging.getLogger()
        logger.info(f"{tag}: {value:.6f} (step {step})")
        
        # Save to JSON file
        self._save_metrics()
    
    def _save_metrics(self):
        """Save metrics to JSON file"""
        metrics_file = self.log_file_path.replace('.txt', '_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)



def save_checkpoint(state, is_best, file_path, file_name='checkpoint.pth.tar'):
    """
    Saves the current state of the model. Does a copy of the file
    in case the model performed better than previously.

    Parameters:
        state (dict): Includes optimizer and model state dictionaries.
        is_best (bool): True if model is best performing model.
        file_path (str): Path to save the file.
        file_name (str): File name with extension (default: checkpoint.pth.tar).
    """

    # Create directory if it doesn't exist
    os.makedirs(file_path, exist_ok=True)
    
    save_path = os.path.join(file_path, file_name)
    torch.save(state, save_path)
    if is_best:
        shutil.copyfile(save_path, os.path.join(file_path, 'model_best.pth.tar'))


def save_task_checkpoint(file_path, task_num):
    """
    Saves the current state of the model for a given task by copying existing checkpoint created by the
    save_checkpoint function.

    Parameters:
        file_path (str): Path to save the file,
        task_num (int): Number of task increment.
    """
    save_path = os.path.join(file_path, 'checkpoint_task_' + str(task_num) + '.pth.tar')
    shutil.copyfile(os.path.join(file_path, 'checkpoint.pth.tar'), save_path)