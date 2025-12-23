import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
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
        patience: int = 10
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
        
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch."""
        from training.metrics import SegmentationMetrics
        
        self.model.train()
        metrics = SegmentationMetrics(self.num_classes)
        
        total_loss = 0.0
        num_batches = 0
        
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
        
        avg_loss = total_loss / num_batches
        return avg_loss, metrics.get_metrics_dict()
    
    def validate_epoch(self):
        """Validate for one epoch."""
        from training.metrics import SegmentationMetrics
        
        self.model.eval()
        metrics = SegmentationMetrics(self.num_classes)
        
        total_loss = 0.0
        num_batches = 0
        
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
        
        avg_loss = total_loss / num_batches
        return avg_loss, metrics.get_metrics_dict()
    
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
        
        for epoch in range(start_epoch, num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 60)
            
            train_loss, train_metrics = self.train_epoch()
            val_loss, val_metrics = self.validate_epoch()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Learning Rate: {current_lr:.2e}")
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Train Acc:  {train_metrics['pixel_accuracy']:.4f} | Val Acc:  {val_metrics['pixel_accuracy']:.4f}")
            print(f"  Train mIoU: {train_metrics['miou']:.4f} | Val mIoU: {val_metrics['miou']:.4f}")
            
            if self.num_classes == 2:
                print(f"  Train IoU [bg, fg]: [{train_metrics['iou_class_0']:.4f}, {train_metrics['iou_class_1']:.4f}]")
                print(f"  Val IoU   [bg, fg]: [{val_metrics['iou_class_0']:.4f}, {val_metrics['iou_class_1']:.4f}]")
            
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
        print("=" * 60)
        
        return self.best_miou

