import sys
import string

import sys
import string
import re

import itertools



def process_large_file(input_file, output_file):
    translator = str.maketrans('', '', string.punctuation)  # Translation table for removing punctuation

    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            line = line.translate(translator)  # Remove punctuation
            words = line.lower().split()  # Convert to lowercase and split into words
            filtered_words = [word for word in words if
                              re.fullmatch(r'[а-щґєіїюяьА-ЩҐЄІЇЮЯЬ]+', word)]  # Keep only Ukrainian letters
            outfile.write("\n".join(filtered_words) + "\n")  # Write each word on a new line


def count_lines(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def process_large_wordlist(input_file, output_file, max_lines=20000):
    """
    Process a large wordlist file efficiently:
    1. Remove empty lines
    2. Limit output to specified number of lines
    3. Use minimal memory

    Args:
        input_file (str): Path to the input wordlist file
        output_file (str): Path to the output processed wordlist file
        max_lines (int): Maximum number of lines to write to output file
    """
    with open(input_file, 'r', encoding='utf-8') as infile, \
            open(output_file, 'w', encoding='utf-8') as outfile:
        # Use a generator to filter out empty lines and limit lines
        processed_lines = itertools.islice(
            (line.strip() for line in infile if line.strip()),
            max_lines
        )

        # Write processed lines to output file
        for line in processed_lines:
            outfile.write(line + '\n')


if __name__ == "__main__":
    # Example usage
    input_path = 'ubertext-words.txt'
    output_path = 'ubertext-words-20k.txt'
    process_large_wordlist(input_path, output_path)
