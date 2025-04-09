# Load model directly
import torch
import tqdm
import transformers
import peft
import pathlib

from src import graph_dataset

# Load model and processor
processor = transformers.AutoProcessor.from_pretrained(
  "Qwen/Qwen2.5-VL-3B-Instruct", device_map="auto", torch_dtype=torch.float16, use_fast=True
)
model = transformers.AutoModelForImageTextToText.from_pretrained(
  "Qwen/Qwen2.5-VL-3B-Instruct",
  device_map="auto",
  torch_dtype=torch.float16,
  use_cache=False,
)
processor.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
model.resize_token_embeddings(len(processor.tokenizer))

#
#
#
# Configure LoRA adapter
lora_config = peft.LoraConfig(
  task_type=peft.TaskType.CAUSAL_LM,
  r=8,  # Rank of the update matrices
  lora_alpha=32,  # Parameter for scaling
  lora_dropout=0.1,  # Dropout probability for LoRA layers
  target_modules=[
    "q_proj",
    "v_proj",
    "k_proj",
    "o_proj",
    # "gate_proj",
    # "up_proj",
    # "down_proj",
  ],  # Which modules to apply LoRA to
  bias="none",
)

# Apply LoRA adapter to the model
model = peft.get_peft_model(model, lora_config)
print("LoRA adapter applied to the model")

# Print trainable parameters info
model.print_trainable_parameters()

#
#
#
# Load datasets and create dataloaders
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, val_loader = graph_dataset.prepare_training_data(processor, batch_size=16, device=device)
print(f"Created dataloaders with {len(train_loader)} training batches and {len(val_loader)} validation batches")

# Set up training parameters
print(f"Using device: {device}")


#
#
#
def forward_pass(batch):
  labels = batch["input_ids"]

  # Remove non-tensor items that shouldn't be passed to the model
  model_inputs = {k: v for k, v in batch.items() if k not in ["image_file", "code_file"]}

  # Forward pass
  outputs = model(**model_inputs, labels=labels)

  return outputs.loss, outputs.logits


def backward_pass(loss, optimizer, scheduler=None, scaler=None):
  optimizer.zero_grad()

  if scaler is not None:
    # Mixed precision backward pass
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    # Clip gradients to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
  else:
    # Standard backward pass
    loss.backward()
    # Clip gradients to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

  # Update learning rate if scheduler is provided
  if scheduler is not None:
    scheduler.step()

  return loss.item()


#
#
#
def train_loop(model, train_loader, optimizer, scheduler, epoch=None, num_epochs=None, scaler=None):
  model.train()
  total_train_loss = 0

  # Create progress bar for training batches
  desc = "Training" if epoch is None else f"Training (Epoch {epoch + 1}/{num_epochs})"
  train_iterator = tqdm.tqdm(train_loader, desc=desc, leave=False)

  for batch in train_iterator:
    # Use mixed precision for forward pass if scaler is provided
    if scaler is not None:
      with torch.amp.autocast("cuda"):
        loss, outputs = forward_pass(batch)
    else:
      loss, outputs = forward_pass(batch)

    loss_value = backward_pass(loss, optimizer, scheduler, scaler)
    total_train_loss += loss_value
    train_iterator.set_postfix({"loss": f"{loss_value:.4f}"})

  return total_train_loss / len(train_loader)


def validate_loop(model, val_loader, epoch=None, num_epochs=None):
  model.eval()
  total_val_loss = 0

  # Create progress bar for validation batches
  desc = "Validation" if epoch is None else f"Validation (Epoch {epoch + 1}/{num_epochs})"
  val_iterator = tqdm.tqdm(val_loader, desc=desc, leave=False)

  with torch.no_grad():
    for batch in val_iterator:
      loss, outputs = forward_pass(batch)
      loss_value = loss.item()
      total_val_loss += loss_value
      val_iterator.set_postfix({"loss": f"{loss_value:.4f}"})

  return total_val_loss / len(val_loader)


#
#
#
if __name__ == "__main__":
  learning_rate = 1e-5
  num_epochs = 10

  # Get the project root directory for saving weights
  project_root = pathlib.Path(__file__).parent.parent

  # Track best model
  best_val_loss = float("inf")
  best_epoch = -1

  optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

  if torch.cuda.is_available():
    # Initialize mixed precision training with updated constructor
    scaler = torch.amp.GradScaler("cuda")
  else:
    scaler = None

  for epoch in tqdm.tqdm(range(num_epochs), desc="Epochs"):
    train_loss = train_loop(model, train_loader, optimizer, scheduler, epoch, num_epochs, scaler)
    val_loss = validate_loop(model, val_loader, epoch, num_epochs)

    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save the best model based on validation loss
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      best_epoch = epoch + 1

    best_model_dir = str(project_root / "weights/qwen2.5-vl-3b-d3js-finetuned-best")

    # Clear cache between epochs
    if torch.cuda.is_available():
      torch.cuda.empty_cache()

  # Save the final fine-tuned model
  output_dir = "weights/qwen2.5-vl-3b-d3js-finetuned"
  model.save_pretrained(output_dir)

  print(f"Final model saved to {output_dir}")
  print(f"Best model was from epoch {best_epoch} with validation loss: {best_val_loss:.4f}")
