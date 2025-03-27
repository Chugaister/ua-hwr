import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import os

from model import CRNN
from dataset import SyntheticHandwritingDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Configuration
IMG_HEIGHT = 128
NUM_CHANNELS = 1
NUM_CLASSES = 33
HIDDEN_SIZE = 256
NUM_LSTM_LAYERS = 2
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "checkpoints"

# Ensure model save directory exists
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Initialize dataset & dataloaders
train_dataset = SyntheticHandwritingDataset("ubertext-words-20k.txt")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                          collate_fn=train_dataset.collate_fn)

# Initialize model
model = CRNN(IMG_HEIGHT, NUM_CHANNELS, NUM_CLASSES, HIDDEN_SIZE, NUM_LSTM_LAYERS).to(DEVICE)

# Loss function (CTC Loss)
criterion = nn.CTCLoss(blank=0)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch_idx, (images, targets, target_lengths) in enumerate(train_loader):
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)
        target_lengths = target_lengths.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)

        log_probs = outputs.log_softmax(2)
        input_lengths = torch.full(size=(outputs.size(0),), fill_value=outputs.size(1), dtype=torch.long).to(DEVICE)

        loss = criterion(log_probs.permute(1, 0, 2), targets, input_lengths, target_lengths)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:  # Log every 10 batches
            logging.info(
                f"Epoch [{epoch + 1}/{EPOCHS}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    logging.info(f"Epoch [{epoch + 1}/{EPOCHS}] Completed - Train Loss: {avg_loss:.4f}")

    # Save model checkpoint
    model_save_file = os.path.join(MODEL_SAVE_PATH, f"crnn_epoch_{epoch + 1}.pth")
    torch.save(model.state_dict(), model_save_file)
    logging.info(f"Model saved at {model_save_file}")

logging.info("Training complete!")

