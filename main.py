import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import CRNN
from dataset import SyntheticHandwritingDataset

# Configuration
IMG_HEIGHT = 128  # Must match the height used in CRNN
NUM_CHANNELS = 1  # Grayscale images
NUM_CLASSES = 33  # Example: 26 letters + blank + special (adjust based on your dataset)
HIDDEN_SIZE = 256
NUM_LSTM_LAYERS = 2
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize dataset & dataloaders
train_dataset = SyntheticHandwritingDataset("ubertext-words-20k.txt")
# val_dataset = SyntheticHandwritingDataset(split="val")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=train_dataset.collate_fn)
# val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

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

    for images, targets, target_lengths in train_loader:
        images = images.to(DEVICE)  # (B, 1, H, W)
        targets = targets.to(DEVICE)  # Flattened ground-truth labels
        target_lengths = target_lengths.to(DEVICE)  # Lengths of each ground-truth sequence

        optimizer.zero_grad()
        outputs = model(images)  # (B, W, num_classes)

        # CTC Loss requires log probabilities
        log_probs = outputs.log_softmax(2)  # (B, W, num_classes)

        # CTC expects input lengths (W) per batch item
        input_lengths = torch.full(size=(outputs.size(0),), fill_value=outputs.size(1), dtype=torch.long).to(DEVICE)

        loss = criterion(log_probs.permute(1, 0, 2), targets, input_lengths, target_lengths)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)  # Prevent exploding gradients
        optimizer.step()
        loss_item = loss.item()
        total_loss += loss_item

    # Validation
    # model.eval()
    # val_loss = 0
    # with torch.no_grad():
    #     for images, targets, target_lengths in val_loader:
    #         images = images.to(DEVICE)
    #         targets = targets.to(DEVICE)
    #         target_lengths = target_lengths.to(DEVICE)
    #
    #         outputs = model(images)
    #         log_probs = outputs.log_softmax(2)
    #
    #         input_lengths = torch.full(size=(outputs.size(0),), fill_value=outputs.size(1), dtype=torch.long).to(DEVICE)
    #         loss = criterion(log_probs.permute(1, 0, 2), targets, input_lengths, target_lengths)
    #         val_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {total_loss/len(train_loader):.4f}") # , Val Loss: {val_loss/len(val_loader):.4f}")

print("Training complete!")
