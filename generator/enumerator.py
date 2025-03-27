import csv

input_file = "updated_dataset.csv"  # Replace with your actual file path
output_file = "glyphs.csv"

with open(input_file, newline='', encoding='utf-8') as infile:
    reader = csv.reader(infile)
    rows = list(reader)

# Insert "id" column header and populate with numbers
rows[0].insert(0, "id")
for i, row in enumerate(rows[1:], start=1):
    row.insert(0, str(i))

with open(output_file, "w", newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    writer.writerows(rows)

print(f"Processed file saved as {output_file}")
