import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, model, train_dataset, val_dataset,device,batch_size=64,
                 epochs=100, learning_rate=0.001, log_interval=20, tb_dir='runs', ckpt_dir='checkpoints'):
        self.device = device
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.log_interval = log_interval
        self.tb_dir = tb_dir  # TensorBoard 日志目录
        self.ckpt_dir = ckpt_dir  # 检查点目录

        # Initialize data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize optimizer and scheduler
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.9)

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.tb_dir)

        # Load checkpoint if it exists
        self.checkpoint_path = os.path.join(self.ckpt_dir, 'checkpoint.pth')
        self.start_epoch = 1
        if os.path.exists(self.checkpoint_path):
            self.load_checkpoint()

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            self.model.train()
            total_loss = 0
            for batch_idx, data in enumerate(self.train_loader):
                data = data.to(self.device)
                self.optimizer.zero_grad()
                loss = self.model.loss(data)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                if batch_idx % self.log_interval == 0:
                    print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}')
                    self.writer.add_scalar('Training Loss', loss.item(), epoch * len(self.train_loader) + batch_idx)
            self.scheduler.step()
            print(f'Epoch {epoch} finished. Average loss: {total_loss / len(self.train_loader)}')
            self.validate()
            self.save_checkpoint(epoch)

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for data in self.val_loader:
                data = data.to(self.device)
                loss = self.model.loss(data)
                total_loss += loss.item()
                self.writer.add_scalar('Validation Loss', loss.item(), self.scheduler.last_epoch)
        average_loss = total_loss / len(self.val_loader)
        print(f'Validation loss: {average_loss}')

    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        torch.save(checkpoint, self.checkpoint_path)

    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    def close_writer(self):
        self.writer.close()