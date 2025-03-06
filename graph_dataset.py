import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
from PIL import Image


class GraphCodeDataset(Dataset):
  def __init__(self, embeddings_file, code_dir, processor, max_length=512):
    """
    Dataset for graph embeddings paired with their corresponding code.

    Args:
        embeddings_file: Path to the .npy file containing image embeddings
        code_dir: Directory containing the D3.js code files
        processor: Qwen processor for encoding the inputs
        max_length: Maximum sequence length for tokenization
    """
    # Load embeddings
    self.embeddings_dict = np.load(embeddings_file, allow_pickle=True).item()
    self.image_files = list(self.embeddings_dict.keys())
    self.processor = processor
    self.max_length = max_length
    self.code_dir = code_dir

    # Map image filenames to code filenames
    self.image_to_code_map = {}
    for img_file in self.image_files:
      # Extract the base name without extension
      base_name = os.path.splitext(img_file)[0]
      code_file = f"{base_name}.js"
      code_path = os.path.join(code_dir, code_file)
      if os.path.exists(code_path):
        self.image_to_code_map[img_file] = code_path

    # Filter to only include images that have corresponding code
    self.valid_images = [img for img in self.image_files if img in self.image_to_code_map]

    print(f"Loaded {len(self.valid_images)} valid image-code pairs")

  def __len__(self):
    return len(self.valid_images)

  def __getitem__(self, idx):
    img_file = self.valid_images[idx]
    code_file = self.image_to_code_map[img_file]

    # Load the original image
    img_path = os.path.join("data", "generated_graphs", img_file)
    image = Image.open(img_path).convert("RGB")

    # Read code content
    with open(code_file, "r") as f:
      code_content = f.read()

    # Create messages for chat template with proper image formatting
    messages = [
      {
        "role": "user",
        "content": [
          {"type": "image", "image": image},
          {"type": "text", "text": "Generate D3.js code for the graph in this image:"},
        ],
      },
      {"role": "assistant", "content": code_content},
    ]

    # Apply chat template
    text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Process inputs using the Qwen processor
    inputs = self.processor(
      text=text,
      padding="max_length",
      max_length=self.max_length,
      truncation=True,
      return_tensors="pt",
    )

    # Remove batch dimension
    for key in inputs:
      if isinstance(inputs[key], torch.Tensor):
        inputs[key] = inputs[key].squeeze(0)

    # Add original filenames for reference
    inputs["image_file"] = img_file
    inputs["code_file"] = code_file

    return inputs


def prepare_training_data(processor, batch_size=4, device="cpu"):
  """
  Prepare training and validation datasets and dataloaders
  """
  # Paths
  embeddings_file = os.path.join("data", "embeddings", "graph_embeddings.npy")
  code_dir = os.path.join("data", "generated_code")

  # Check if embeddings exist, if not generate them
  if not os.path.exists(embeddings_file):
    print("Embeddings file not found. Generating embeddings...")
    from image_embeddings import generate_embeddings

    generate_embeddings()

  # Create dataset
  dataset = GraphCodeDataset(embeddings_file, code_dir, processor)

  # Split into train and validation sets (80/20)
  train_size = int(0.8 * len(dataset))
  val_size = len(dataset) - train_size
  train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

  # Create a collate function to move data to the correct device
  def collate_fn(batch):
    collated_batch = {}

    for key in batch[0].keys():
      if key in ["image_file", "code_file"]:
        # Handle string values like filenames - don't try to stack them
        collated_batch[key] = [item[key] for item in batch]
      else:
        # For tensors, stack them and move to device
        collated_batch[key] = torch.stack([item[key] for item in batch]).to(device)

    return collated_batch

  # Create data loaders with collate function
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

  return train_loader, val_loader


def create_fine_tuning_dataset(output_dir="data/fine_tuning"):
  """
  Create and save fine-tuning dataset in a format suitable for HuggingFace
  """
  # Paths
  embeddings_file = os.path.join("data", "embeddings", "graph_embeddings.npy")
  code_dir = os.path.join("data", "generated_code")

  # Check if embeddings exist, if not generate them
  if not os.path.exists(embeddings_file):
    print("Embeddings file not found. Generating embeddings...")
    from image_embeddings import generate_embeddings

    generate_embeddings()

  # Load embeddings
  embeddings_dict = np.load(embeddings_file, allow_pickle=True).item()

  # Create output directory
  os.makedirs(output_dir, exist_ok=True)

  # Create dataset in JSONL format
  train_data = []

  for img_file, embedding in embeddings_dict.items():
    # Extract the base name without extension
    base_name = os.path.splitext(img_file)[0]
    code_file = f"{base_name}.js"
    code_path = os.path.join(code_dir, code_file)

    if os.path.exists(code_path):
      with open(code_path, "r") as f:
        code_content = f.read()

      # Create a sample with instruction, input (embedding), and output (code)
      sample = {
        "instruction": "Generate D3.js code for the graph in this image.",
        "input": embedding.tolist(),  # Convert numpy array to list for JSON serialization
        "output": code_content,
      }

      train_data.append(sample)

  # Split into train and validation sets (80/20)
  train_size = int(0.8 * len(train_data))
  train_samples = train_data[:train_size]
  val_samples = train_data[train_size:]

  # Save datasets
  with open(os.path.join(output_dir, "train.json"), "w") as f:
    json.dump(train_samples, f)

  with open(os.path.join(output_dir, "validation.json"), "w") as f:
    json.dump(val_samples, f)

  print(
    f"Created fine-tuning dataset with {len(train_samples)} training samples and {len(val_samples)} validation samples"
  )
  return len(train_samples), len(val_samples)


if __name__ == "__main__":
  create_fine_tuning_dataset()
