import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random
import numpy as np
from generator.generator import HandwritingGenerator  # Import your generator


class SyntheticHandwritingDataset(Dataset):
    """
    Custom dataset for OCR training with CTC loss.
    Generates handwritten images using `HandwritingGenerator` from Ukrainian words.
    """

    char_to_idx = {
        "а": 1, "б": 2, "в": 3, "г": 4, "ґ": 5, "д": 6, "е": 7, "є": 8, "ж": 9, "з": 10, "и": 11, "і": 12,
        "ї": 13, "й": 14, "к": 15, "л": 16, "м": 17, "н": 18, "о": 19, "п": 20, "р": 21, "с": 22, "т": 23,
        "у": 24, "ф": 25, "х": 26, "ц": 27, "ч": 28, "ш": 29, "щ": 30, "ь": 31, "ю": 32, "я": 33
    }

    def __init__(self, words_path, img_height=64):
        """
        Args:
            words_path (str): Path to a text file containing Ukrainian words (one per line).
            handwriting_generator (HandwritingGenerator): Instance of HandwritingGenerator.
            char_to_idx (dict): Mapping of characters to indices.
            img_height (int): Fixed height for resizing images.
        """
        self.handwriting_generator = HandwritingGenerator("generator/glyphs.csv")

        # Load words
        with open(words_path, "r", encoding="utf-8") as f:
            self.words = []
            for _ in range(self.__len__()):
                self.words.append(f.readline().strip())

        # Image transformations
        self.transform = transforms.Compose([
            transforms.Grayscale(),  # Convert to 1 channel
            transforms.Resize((img_height, img_height * 6)),  # Resize with fixed height
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize
        ])

    # @property
    # def num_chars(self) -> int:
    #     return len(self.char_to_idx)

    def __len__(self):
        # return len(self.words)
        return 20000

    def __getitem__(self, idx):
        """
        Returns:
            image (Tensor): Shape (1, H, W)
            target (Tensor): Encoded text sequence
            target_length (Tensor): Length of target sequence
        """
        word = self.words[idx]

        # Generate handwriting image
        image = self.handwriting_generator.generate(word)

        # Apply transformations
        image = self.transform(image)

        # Convert text to indices
        target = torch.tensor([self.char_to_idx[c] for c in word], dtype=torch.long)
        target_length = torch.tensor(len(target), dtype=torch.long)

        return image, target, target_length

    @staticmethod
    def collate_fn(batch):
        images, labels, lengths = zip(*batch)  # Unpack three elements: images, labels, lengths

        # Get the max width and height of the images in the batch
        max_height = max(image.shape[1] for image in images)  # Height is the second dimension (H)
        max_width = max(image.shape[2] for image in images)  # Width is the third dimension (W)

        # Padding images to make them all the same size
        padded_images = []
        for image in images:
            # Create a new tensor of the correct size, filled with 255 (white)
            padded_image = torch.ones((1, max_height, max_width), dtype=torch.float32) * 255.0
            # Copy the original image into the padded tensor
            padded_image[:, :image.shape[1], :image.shape[2]] = image
            padded_images.append(padded_image)

        # Pad the labels (sequences of integers) to the same length
        max_label_length = max(len(label) for label in labels)
        padded_labels = []
        for label in labels:
            # Pad the label with zeros
            padding = torch.zeros(max_label_length - len(label), dtype=torch.long)
            padded_label = torch.cat([label, padding])  # Concatenate label with padding
            padded_labels.append(padded_label)

        # Stack the padded images into a single tensor
        padded_images = torch.stack(padded_images)

        # Convert lengths to a tensor
        lengths = torch.tensor(lengths, dtype=torch.long)

        # Return padded images, padded labels as a tensor, and lengths
        return padded_images, torch.stack(padded_labels), lengths



