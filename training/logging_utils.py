"""Utilities for setting up training execution logging."""

import sys
import os


class DualOutput:
    """Write to both file and terminal simultaneously."""
    
    def __init__(self, file_path: str, terminal):
        self.file = open(file_path, 'w')
        self.terminal = terminal
    
    def write(self, message):
        try:
            self.terminal.write(message)
        except UnicodeEncodeError:
            # Handle Windows console encoding issues
            self.terminal.write(message.encode('ascii', 'replace').decode('ascii'))
        self.file.write(message)
        self.file.flush()
    
    def flush(self):
        self.terminal.flush()
        self.file.flush()
    
    def close(self):
        self.file.close()


def setup_logging(run_dir: str, log_filename: str = "training.log"):
    """
    Setup logging to both console and file using stdout redirection.
    
    Redirects sys.stdout to write to both terminal and a log file.
    This captures ALL output including print statements, tqdm progress bars, etc.
    
    Args:
        run_dir: Directory to save the log file
        log_filename: Name of the log file (default: "training.log")
        
    Returns:
        Tuple of (log_file_path, original_stdout)
        
    Usage:
        log_file, original_stdout = setup_logging(run_dir)
        try:
            print("Training...")  # Goes to both console and file
        finally:
            sys.stdout.close()
            sys.stdout = original_stdout
    """
    log_file = os.path.join(run_dir, log_filename)
    
    original_stdout = sys.stdout
    sys.stdout = DualOutput(log_file, original_stdout)
    
    return log_file, original_stdout

