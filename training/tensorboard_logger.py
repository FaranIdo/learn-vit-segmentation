from torch.utils.tensorboard import SummaryWriter
import os


class TensorBoardLogger:
    """
    Wrapper for TensorBoard SummaryWriter with simplified API.
    
    Usage:
        logger = TensorBoardLogger(log_dir="runs/experiment1")
        logger.log_scalar("loss", 0.5, step=10)
        logger.log_scalars("metrics", {"train": 0.8, "val": 0.7}, step=10)
        logger.close()
    """
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log a single scalar value."""
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        """
        Log multiple scalars in the same plot.
        
        Args:
            main_tag: Parent tag (e.g., "Loss")
            tag_scalar_dict: Dict of {subtag: value} (e.g., {"train": 0.5, "val": 0.3})
            step: Training step/epoch
        """
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_learning_rate(self, lr: float, step: int):
        """Log learning rate."""
        self.log_scalar("Learning_Rate", lr, step)
    
    def log_image(self, tag: str, image, step: int):
        """
        Log an image tensor.
        
        Args:
            tag: Image tag
            image: Image tensor [C, H, W] or [B, C, H, W]
            step: Training step/epoch
        """
        self.writer.add_image(tag, image, step)
    
    def log_images(self, tag: str, images, step: int, max_images: int = 16):
        """
        Log a batch of images as a grid.
        
        Args:
            tag: Image tag
            images: Image tensor [B, C, H, W]
            step: Training step/epoch
            max_images: Maximum number of images to display
        """
        if images.shape[0] > max_images:
            images = images[:max_images]
        
        self.writer.add_images(tag, images, step)
    
    def log_histogram(self, tag: str, values, step: int):
        """Log a histogram of values."""
        self.writer.add_histogram(tag, values, step)
    
    def log_text(self, tag: str, text: str, step: int):
        """Log text."""
        self.writer.add_text(tag, text, step)
    
    def flush(self):
        """Flush pending events to disk."""
        self.writer.flush()
    
    def close(self):
        """Close the writer."""
        self.writer.close()

