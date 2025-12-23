import sys
import os
from datetime import datetime


class ExecutionLogger:
    """
    Logger that captures all stdout and saves to both console and file.
    
    Usage:
        logger = ExecutionLogger("checkpoints/run_2024-12-23_22-45-30")
        # All print statements are now logged to file
        logger.close()
    """
    
    def __init__(self, log_dir: str, log_filename: str = "training.log"):
        self.log_dir = log_dir
        self.log_path = os.path.join(log_dir, log_filename)
        
        os.makedirs(log_dir, exist_ok=True)
        
        self.log_file = open(self.log_path, 'w', encoding='utf-8')
        self.terminal = sys.stdout
        
        sys.stdout = self
        
        self._write_header()
    
    def _write_header(self):
        """Write log header with timestamp."""
        header = f"Training Execution Log\n"
        header += f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += f"Log file: {os.path.abspath(self.log_path)}\n"
        header += "=" * 60 + "\n\n"
        
        self.terminal.write(header)
        self.log_file.write(header)
        self.log_file.flush()
    
    def write(self, message):
        """Write message to both terminal and file."""
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
    
    def flush(self):
        """Flush both terminal and file."""
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        """Close the log file and restore stdout."""
        footer = f"\n\nTraining Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        self.terminal.write(footer)
        self.log_file.write(footer)
        
        sys.stdout = self.terminal
        self.log_file.close()
    
    def get_log_path(self) -> str:
        """Get absolute path to log file."""
        return os.path.abspath(self.log_path)

