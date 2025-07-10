from datasets import load_dataset
import os

# Load the dataset
dataset = load_dataset("knkarthick/samsum")

# Make sure output directory exists
output_dir = "samsum_data"
os.makedirs(output_dir, exist_ok=True)

# Save each split to a JSON file in the current folder
for split in dataset.keys():
    output_path = os.path.join(output_dir, f"{split}.json")
    dataset[split].to_json(output_path, orient="records", lines=True)
    print(f"Saved {split} split to {output_path}")
    
