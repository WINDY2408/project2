import os
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import ignite
from ignite.engine import Events, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.handlers.param_scheduler import CosineAnnealingScheduler
from ignite.contrib.handlers import ProgressBar
from ignite.handlers.tensorboard_logger import TensorboardLogger, OutputHandler
from PIL import Image
from datetime import datetime
from termcolor import colored
import time
from sklearn.model_selection import StratifiedGroupKFold
import random

# Constants and Configuration
class Config:
    DATA_DIR = Path("data/state-farm")
    IMGS_DIR = DATA_DIR / "imgs"
    TRAIN_DIR = IMGS_DIR / "train"
    TEST_DIR = IMGS_DIR / "test"
    DRIVER_CSV = DATA_DIR / "driver_imgs_list.csv"
    SAMPLE_SUB_CSV = DATA_DIR / "sample_submission.csv"
    MODEL_DIR = Path("models")
    LOGS_DIR = Path("logs")
    BATCH_SIZE = 64
    NUM_EPOCHS = 300
    INITIAL_LR = 0.01
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42
    NUM_CLASSES = 10
    NUM_WORKERS = 4
    
def set_seed(seed=Config.SEED):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels"""
    
    COLORS = {
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red',
        'DEBUG': 'blue'
    }

    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = colored(f'{levelname:8}', self.COLORS[levelname])
            
        if "Epoch" in record.msg:
            msg_parts = record.msg.split()
            for i, part in enumerate(msg_parts):
                if "Accuracy:" in part or "Acc:" in part:
                    msg_parts[i] = colored(f"Acc: {float(msg_parts[i+1]):.4f}", 'cyan')
                    msg_parts[i+1] = ''
                elif "Loss:" in part:
                    msg_parts[i] = colored(f"Loss: {float(msg_parts[i+1]):.4f}", 'magenta')
                    msg_parts[i+1] = ''
            record.msg = ' '.join(filter(None, msg_parts))
            
        return super().format(record)

class TrainingLogger:
    """Custom logger for training progress"""
    
    def __init__(self, name='training'):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        
        console_handler = logging.StreamHandler()
        formatter = ColoredFormatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        self.epoch_start_time = None
        self.training_start_time = None
        
    def start_training(self):
        self.training_start_time = time.time()
        self.logger.info(colored("Starting training...", 'green'))
        
    def start_epoch(self, epoch):
        self.epoch_start_time = time.time()
        self.logger.info(colored(f"Epoch [{epoch}/{Config.NUM_EPOCHS}]", 'yellow'))
        
    def log_metrics(self, phase, epoch, accuracy, loss):
        time_taken = time.time() - (self.epoch_start_time if phase == "Training" else self.training_start_time)
        self.logger.info(
            f"{phase:9} Epoch [{epoch:2d}] "
            f"Acc: {accuracy:.4f} Loss: {loss:.4f} "
            f"Time: {time_taken:.2f}s"
        )

class DistractedDriverDataset(Dataset):
    """Dataset class for training and validation data"""
    
    def __init__(self, data, labels, transform=None):
        self.data = data.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]
        label = self.labels.iloc[idx]
        img_path = Config.TRAIN_DIR / label / img_name
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, int(label.replace('c', ''))
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            return None

class DistractedDriverTestDataset(Dataset):
    """Dataset class for test data"""
    
    def __init__(self, img_names, transform=None):
        self.img_names = img_names
        self.transform = transform
        
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        img_name = self.img_names.iloc[idx]
        img_path = Config.TEST_DIR / img_name
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, 0
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            return None

class DistractedDriverModel(nn.Module):
    """Model architecture using EfficientNet-B0"""
    
    def __init__(self, num_classes=Config.NUM_CLASSES):
        super(DistractedDriverModel, self).__init__()
        self.base_model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(self.base_model.classifier[1].in_features, num_classes)
        )
        
    def forward(self, x):
        return self.base_model(x)

class DataModule:
    """Data module for handling all data-related operations"""
    
    @staticmethod
    def get_transforms():
        train_transform = transforms.Compose([
            transforms.Resize((240, 240)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2))
        ])
        
        val_test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        return train_transform, val_test_transform
    
    @staticmethod
    def get_data_loaders():
        df = pd.read_csv(Config.DRIVER_CSV)
        df['label'] = df['classname']
        df['filepath'] = df['img']
        
        # Split data using StratifiedGroupKFold
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=Config.SEED)
        train_idx, val_idx = next(skf.split(df['img'], df['label'], groups=df['subject']))
        
        train_data = df.iloc[train_idx]
        val_data = df.iloc[val_idx]
        
        train_transform, val_test_transform = DataModule.get_transforms()
        
        # Create datasets
        train_dataset = DistractedDriverDataset(
            train_data['img'], train_data['label'], transform=train_transform
        )
        val_dataset = DistractedDriverDataset(
            val_data['img'], val_data['label'], transform=val_test_transform
        )
        test_df = pd.read_csv(Config.SAMPLE_SUB_CSV)
        test_dataset = DistractedDriverTestDataset(
            test_df['img'], transform=val_test_transform
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader

class Trainer:
    """Training class handling the training loop and related operations"""
    
    def __init__(self, model, train_loader, val_loader, logger):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger
        self.device = Config.DEVICE
        
        # Initialize optimizer and criterion
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=Config.INITIAL_LR,
            momentum=0.9,
            weight_decay=1e-4
        )
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = torch.cuda.amp.GradScaler()
        
        # Initialize trainer and evaluator
        self.trainer = self._create_trainer()
        self.evaluator = self._create_evaluator()
        
        # Setup directories
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_dir = Config.MODEL_DIR / self.run_id
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup handlers
        self._setup_handlers()
    
    def _create_trainer(self):
        def train_step(engine, batch):
            self.model.train()
            self.optimizer.zero_grad()
            images, labels = batch
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            return loss.item()
        
        return ignite.engine.Engine(train_step)
    
    def _create_evaluator(self):
        def eval_step(engine, batch):
            self.model.eval()
            with torch.no_grad():
                images, labels = batch
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                outputs = self.model(images)
                return outputs, labels
        
        evaluator = ignite.engine.Engine(eval_step)
        
        Accuracy().attach(evaluator, 'accuracy')
        Loss(self.criterion).attach(evaluator, 'loss')
        
        return evaluator

    def _setup_handlers(self):
        # Learning rate scheduler
        scheduler = CosineAnnealingScheduler(
            self.optimizer,
            'lr',
            start_value=Config.INITIAL_LR,
            end_value=0.0001,
            cycle_size=Config.NUM_EPOCHS * len(self.train_loader)
        )
        self.trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
        
        # Progress bar
        pbar = ProgressBar()
        pbar.attach(self.trainer, output_transform=lambda x: {'loss': x})
        
        # TensorBoard logger
        tb_logger = TensorboardLogger(log_dir=Config.LOGS_DIR)
        
        # Model checkpointing
        best_model_handler = ModelCheckpoint(
            self.checkpoint_dir,
            'best',
            n_saved=3,
            require_empty=False,
            score_function=lambda engine: engine.state.metrics['accuracy'],
            score_name="val_accuracy",
            global_step_transform=lambda *_: self.trainer.state.epoch
        )
        self.evaluator.add_event_handler(
            Events.EPOCH_COMPLETED,
            best_model_handler,
            {'model': self.model}
        )
        
        # Early stopping
        early_stopping = EarlyStopping(
            patience=100,
            score_function=lambda engine: engine.state.metrics['accuracy'],
            trainer=self.trainer
        )
        self.evaluator.add_event_handler(Events.COMPLETED, early_stopping)
        
        # Gradual unfreezing
        def unfreeze_layers(engine):
            epoch = engine.state.epoch
            if epoch == 5:
                for param in self.model.base_model.features[7:].parameters():
                    param.requires_grad = True
                self.logger.logger.info(colored("Unfroze layers from block 7 onwards", 'green'))
            elif epoch == 10:
                for param in self.model.base_model.features[5:].parameters():
                    param.requires_grad = True
                self.logger.logger.info(colored("Unfroze layers from block 5 onwards", 'green'))
        
        self.trainer.add_event_handler(Events.EPOCH_STARTED, unfreeze_layers)
        
        # Attach event handlers for logging
        @self.trainer.on(Events.EPOCH_STARTED)
        def log_epoch_start(engine):
            self.logger.start_epoch(engine.state.epoch)
        
        @self.trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(engine):
            self.evaluator.run(self.train_loader)
            metrics = self.evaluator.state.metrics
            self.logger.log_metrics(
                "Training",
                engine.state.epoch,
                metrics['accuracy'],
                metrics['loss']
            )
            
            self.evaluator.run(self.val_loader)
            metrics = self.evaluator.state.metrics
            self.logger.log_metrics(
                "Validation",
                engine.state.epoch,
                metrics['accuracy'],
                metrics['loss']
            )
            
            # Log to TensorBoard
            tb_logger.attach(
                self.evaluator,
                log_handler=OutputHandler(
                    tag="validation",
                    metric_names=['accuracy', 'loss'],
                    global_step_transform=lambda *_: engine.state.epoch
                ),
                event_name=Events.EPOCH_COMPLETED
            )
    
    def train(self):
        self.logger.start_training()
        self.trainer.run(self.train_loader, max_epochs=Config.NUM_EPOCHS)
        return self.checkpoint_dir

def create_submission(model, test_loader, submission_path, logger):
    """Create submission file from model predictions"""
    model.eval()
    predictions = []
    filenames = []
    
    logger.logger.info(colored("Creating submission file...", 'green'))
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(Config.DEVICE)
            try:
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                predictions.extend(probs.cpu().numpy())
                filenames.extend(test_loader.dataset.img_names[len(predictions)-len(images):len(predictions)].tolist())
            except Exception as e:
                logger.logger.error(f"Error during prediction: {str(e)}")
                continue
    
    submission_df = pd.DataFrame(predictions, columns=[f'c{i}' for i in range(Config.NUM_CLASSES)])
    submission_df.insert(0, 'img', filenames)
    
    try:
        submission_df.to_csv(submission_path, index=False)
        logger.logger.info(colored(f"Submission saved to {submission_path}", 'green'))
    except Exception as e:
        logger.logger.error(f"Error saving submission file: {str(e)}")

def check_requirements():
    """Check if all required paths and dependencies are available"""
    required_paths = [
        Config.TRAIN_DIR,
        Config.TEST_DIR,
        Config.DRIVER_CSV,
        Config.SAMPLE_SUB_CSV
    ]
    
    missing_paths = []
    for path in required_paths:
        if not path.exists():
            missing_paths.append(path)
    
    return missing_paths

def main():
    """Main training pipeline"""
    # Initialize logger
    training_logger = TrainingLogger()
    training_logger.logger.info(colored(f"Using device: {Config.DEVICE}", 'green'))
    
    # Set random seed
    set_seed()
    
    # Check requirements
    missing_paths = check_requirements()
    if missing_paths:
        for path in missing_paths:
            training_logger.logger.error(f"Required path not found: {path}")
        sys.exit(1)
    
    # Create necessary directories
    Config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    Config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # Prepare data
        training_logger.logger.info(colored("Preparing data loaders...", 'green'))
        train_loader, val_loader, test_loader = DataModule.get_data_loaders()
        
        # Initialize model
        model = DistractedDriverModel().to(Config.DEVICE)
        
        # Freeze initial layers
        for param in model.base_model.features.parameters():
            param.requires_grad = False
        
        # Create trainer and start training
        trainer = Trainer(model, train_loader, val_loader, training_logger)
        checkpoint_dir = trainer.train()
        
        # Load best model
        best_model_path = list(checkpoint_dir.glob('best_model_*.pt'))[-1]
        best_model = DistractedDriverModel().to(Config.DEVICE)
        
        try:
            state_dict = torch.load(best_model_path)
            if isinstance(state_dict, dict) and 'model' in state_dict:
                best_model.load_state_dict(state_dict['model'])
            else:
                best_model.load_state_dict(state_dict)
            training_logger.logger.info(colored(f"Loaded best model from {best_model_path}", 'green'))
        except Exception as e:
            training_logger.logger.error(f"Error loading best model: {str(e)}")
            sys.exit(1)
        
        # Create submission
        submission_path = checkpoint_dir / 'submission.csv'
        create_submission(best_model, test_loader, submission_path, training_logger)
        
        training_logger.logger.info(colored("Training completed successfully!", 'green'))
        
    except Exception as e:
        training_logger.logger.error(f"Error during training: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()