import cv2
import pandas as pd
import numpy as np
from typing import List
from PIL import Image
from time import perf_counter
from os import path


class WordGeneratorError(Exception):
    pass


class HandwritingGenerator:

    def __init__(self, dataset_path: str):
        self.dataset = pd.read_csv(dataset_path)

    def validate_text(self, text: str) -> bool:
        return all('а' <= ch <= 'я' or 'A' <= ch <= 'Я' or ch == 'є' or ch == 'і' or ch == 'ї' or ch == 'ґ' or ch == ' ' for ch in text)

    def get_glyphs_ids(self, text: str) -> List[int]:
        glyph_ids = []
        for ch in text:
            row = self.dataset[(self.dataset['label'] == ch)]
            if row.empty:
                raise WordGeneratorError(f"Glyph not found for character: {ch}")
            glyph_ids.append(row.sample(n=1).iloc[0]["id"])  # Randomly select one glyph
        return glyph_ids

    def resize_with_padding(self, img: Image.Image, target_height: int):
        aspect_ratio = img.width / img.height
        new_width = int(target_height * aspect_ratio)
        img = img.resize((new_width, target_height), Image.BICUBIC)
        return img

    def generate_by_ids(self, glyph_ids: list[int]) -> Image:
        glyphs_data = self.dataset.set_index("id").loc[glyph_ids].reset_index()
        glyph_images = []

        for _, glyph in glyphs_data.iterrows():  # ✅ Fix: Iterate correctly over rows
            img_path = path.join(path.dirname(__file__), glyph['filename'])
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

            if img is None:
                raise WordGeneratorError(f"Could not load image: {img_path}")

            if img.shape[-1] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

            glyph_images.append((glyph, img))

        bottom_align = 200
        # Calculate output image size
        total_width = sum(glyph["bright"] - glyph["bleft"] for glyph, _ in glyph_images)
        total_width += glyph_images[0][0]["bleft"] + glyph_images[-1][1].shape[1] - glyph_images[-1][0]["bright"]
        # total_width = 2500

        max_height = max(bottom_align - int(glyph["bbottom"]) + img.shape[0] for glyph, img in glyph_images)
        # max_height = 400

        # Create blank image
        output_image = np.ones((int(max_height), int(total_width), 4), dtype=np.uint8) * 255

        x_offset = int(glyph_images[0][0]["bleft"])
        for glyph, img in glyph_images:
            y_offset = bottom_align - int(glyph["bbottom"])
            x_offset -= int(glyph["bleft"])
            black_pixels_mask = np.all(img[:, :, :3] == 0, axis=-1)
            output_image[y_offset:y_offset + img.shape[0], x_offset:x_offset + img.shape[1]] = np.where(
                black_pixels_mask[:, :, np.newaxis], img,
                output_image[y_offset:y_offset + img.shape[0], x_offset:x_offset + img.shape[1]]
            )
            x_offset += int(glyph["bright"])

        img = Image.fromarray(output_image)

        return self.resize_with_padding(img, 128)

    def generate(self, word: str) -> Image:
        if not self.validate_text(word):
            raise WordGeneratorError("The word is not valid. Use only Ukrainian lowercase letters without punctuation.")

        glyphs_ids = self.get_glyphs_ids(word)

        return self.generate_by_ids(glyphs_ids)


if __name__ == "__main__":
    # Example usage
    generator = HandwritingGenerator()

    start = perf_counter()
    word_image = generator.generate("ж"*32)
    print(perf_counter() - start)
    word_image.save("generated_word.png")
