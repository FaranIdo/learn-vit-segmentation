import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time
from pathlib import Path


class Trainer:
    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion,
        optimizer,
        scheduler,
        device,
        num_classes: int,
        checkpoint_dir: str = "./checkpoints",
        patience: int = 10,
        use_tensorboard: bool = True
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_classes = num_classes
        self.checkpoint_dir = checkpoint_dir
        self.patience = patience

        self.best_miou = 0.0
        self.epochs_without_improvement = 0
        
        self.start_time = None
        self.epoch_times = []

        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        if use_tensorboard:
            from training.tensorboard_logger import TensorBoardLogger
            self.tensorboard = TensorBoardLogger(
                log_dir=os.path.join(checkpoint_dir, "tensorboard")
            )
        else:
            self.tensorboard = None

    def train_epoch(self):
        """Train for one epoch."""
        from training.metrics import SegmentationMetrics

        self.model.train()
        metrics = SegmentationMetrics(self.num_classes)

        total_loss = 0.0
        num_batches = 0
        
        start_time = time.time()

        pbar = tqdm(self.train_loader, desc="Training")
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, masks)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            preds = torch.argmax(logits, dim=1)
            metrics.update(preds, masks)

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'miou': f'{metrics.get_miou():.4f}'
            })
        
        elapsed_time = time.time() - start_time

        avg_loss = total_loss / num_batches
        return avg_loss, metrics.get_metrics_dict(), elapsed_time

    def validate_epoch(self):
        """Validate for one epoch."""
        from training.metrics import SegmentationMetrics

        self.model.eval()
        metrics = SegmentationMetrics(self.num_classes)

        total_loss = 0.0
        num_batches = 0
        
        start_time = time.time()

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)

                logits = self.model(images)
                loss = self.criterion(logits, masks)

                preds = torch.argmax(logits, dim=1)
                metrics.update(preds, masks)

                total_loss += loss.item()
                num_batches += 1

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'miou': f'{metrics.get_miou():.4f}'
                })
        
        elapsed_time = time.time() - start_time

        avg_loss = total_loss / num_batches
        return avg_loss, metrics.get_metrics_dict(), elapsed_time

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_miou': self.best_miou,
            'metrics': metrics
        }

        checkpoint_path = os.path.join(self.checkpoint_dir, 'last_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model (mIoU: {self.best_miou:.4f})")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_miou = checkpoint['best_miou']

        return checkpoint['epoch']

    def should_stop_early(self) -> bool:
        """Check if training should stop early."""
        return self.epochs_without_improvement >= self.patience
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds into human-readable time string."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

    def train(self, num_epochs: int, start_epoch: int = 0):
        """Train for multiple epochs."""
        print("=" * 60)
        print("Starting Training")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Num epochs: {num_epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"Early stopping patience: {self.patience}")
        print("=" * 60)
        
        self.start_time = time.time()

        for epoch in range(start_epoch, num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 60)

            train_loss, train_metrics, train_time = self.train_epoch()
            val_loss, val_metrics, val_time = self.validate_epoch()
            
            epoch_time = train_time + val_time
            self.epoch_times.append(epoch_time)

            current_lr = self.optimizer.param_groups[0]['lr']

            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Learning Rate: {current_lr:.2e} ({current_lr:.6f})")
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Train Acc:  {train_metrics['pixel_accuracy']:.4f} | Val Acc:  {val_metrics['pixel_accuracy']:.4f}")
            print(f"  Train mIoU: {train_metrics['miou']:.4f} | Val mIoU: {val_metrics['miou']:.4f}")

            if self.num_classes == 2:
                print(f"  Train IoU [bg, fg]: [{train_metrics['iou_class_0']:.4f}, {train_metrics['iou_class_1']:.4f}]")
                print(f"  Val IoU   [bg, fg]: [{val_metrics['iou_class_0']:.4f}, {val_metrics['iou_class_1']:.4f}]")
            
            total_elapsed = time.time() - self.start_time
            avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
            remaining_epochs = num_epochs - (epoch + 1)
            estimated_remaining = avg_epoch_time * remaining_epochs
            estimated_completion_time = time.time() + estimated_remaining
            
            print(f"\nTiming:")
            print(f"  Epoch time: {self._format_time(epoch_time)} (Train: {self._format_time(train_time)} | Val: {self._format_time(val_time)})")
            print(f"  Avg epoch time: {self._format_time(avg_epoch_time)}")
            
            if remaining_epochs > 0:
                print(f"  Estimated remaining: {self._format_time(estimated_remaining)} ({remaining_epochs} epochs × {self._format_time(avg_epoch_time)})")
                completion_struct = time.localtime(estimated_completion_time)
                completion_str = time.strftime("%I:%M %p", completion_struct)
                print(f"  Estimated completion: {completion_str}")
            
            print(f"  Total elapsed: {self._format_time(total_elapsed)}")
            
            if self.tensorboard:
                self.tensorboard.log_scalars("Loss", {
                    "train": train_loss,
                    "val": val_loss
                }, epoch + 1)
                
                self.tensorboard.log_scalars("mIoU", {
                    "train": train_metrics['miou'],
                    "val": val_metrics['miou']
                }, epoch + 1)
                
                self.tensorboard.log_scalars("Pixel_Accuracy", {
                    "train": train_metrics['pixel_accuracy'],
                    "val": val_metrics['pixel_accuracy']
                }, epoch + 1)
                
                for c in range(self.num_classes):
                    self.tensorboard.log_scalars(f"IoU_Class_{c}", {
                        "train": train_metrics[f'iou_class_{c}'],
                        "val": val_metrics[f'iou_class_{c}']
                    }, epoch + 1)
                
                self.tensorboard.log_learning_rate(current_lr, epoch + 1)

            val_miou = val_metrics['miou']
            is_best = val_miou > self.best_miou

            if is_best:
                self.best_miou = val_miou
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            self.save_checkpoint(epoch + 1, val_metrics, is_best=is_best)

            if self.should_stop_early():
                print(f"\n⚠ Early stopping triggered! No improvement for {self.patience} epochs.")
                print(f"Best mIoU: {self.best_miou:.4f}")
                break

            self.scheduler.step()

        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Best Validation mIoU: {self.best_miou:.4f}")
        final_time = time.time() - self.start_time
        print(f"Total training time: {self._format_time(final_time)}")
        print("=" * 60)
        
        if self.tensorboard:
            self.tensorboard.close()

        return self.best_miou
