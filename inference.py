import torch
from PIL import Image
import argparse
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
import os


def load_model(model_path):
  """
  Load the fine-tuned model and processor
  """
  # Check if model path exists
  if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model path {model_path} does not exist. Please run training first.")

  # Load processor from the base model, not the fine-tuned path
  processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
  processor.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

  # Load base model
  base_model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", device_map="auto", torch_dtype=torch.float16
  )
  base_model.resize_token_embeddings(len(processor.tokenizer))

  # Load fine-tuned model
  model = PeftModel.from_pretrained(base_model, model_path)
  model.eval()

  return model, processor


def generate_d3_code(model, processor, image_path, output_path=None, max_length=1024):
  """
  Generate D3.js code for a graph image
  """
  # Load and process the image
  image = Image.open(image_path).convert("RGB")

  # Create messages for chat template with proper image formatting
  messages = [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "Generate D3.js code for the graph in this image:"},
      ],
    }
  ]

  # Apply chat template
  text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

  # Process inputs
  inputs = processor(text=text, return_tensors="pt").to(model.device)

  # Generate code
  with torch.no_grad():
    outputs = model.generate(**inputs, max_length=max_length, do_sample=False, temperature=0.1, num_beams=3)

  # Decode the generated text
  generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

  # Extract just the code part (remove any assistant prefixes)
  if "assistant" in generated_text.lower():
    code = generated_text.split("assistant")[-1].strip()
    if code.startswith(":"):
      code = code[1:].strip()
  else:
    code = generated_text

  # Create a complete HTML page with the generated D3.js code
  html_template = f"""<!DOCTYPE html>
<html>
  <head>
    <meta charset="UTF-8" />
    <title>D3.js Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
      body {{
        font-family: Arial, sans-serif;
      }}
      .chart-container {{
        margin: 20px;
      }}
    </style>
  </head>
  <body>
    <div class="chart-container">
      <h2>Generated Visualization</h2>
      <svg id="chart" width="800" height="500"></svg>
    </div>
    <script>
      // D3.js code
      {code}
    </script>
  </body>
</html>"""

  # Save to file if output path is provided
  if output_path:
    # If output doesn't end with .html, add .html extension
    if not output_path.endswith(".html"):
      output_path = f"{output_path}.html"

    with open(output_path, "w") as f:
      f.write(html_template)
    print(f"Complete HTML with D3.js code saved to {output_path}")

  return code


def main():
  parser = argparse.ArgumentParser(description="Generate D3.js code from graph images")
  parser.add_argument("--image", type=str, required=True, help="Path to the graph image")
  parser.add_argument(
    "--model", type=str, default="models/qwen2.5-vl-3b-d3js-finetuned-best", help="Path to the fine-tuned model"
  )
  parser.add_argument(
    "--output",
    type=str,
    default="./inference_results/graph.html",
    help="Path to save the generated HTML with D3.js code (optional)",
  )
  parser.add_argument("--max_length", type=int, default=2048, help="Maximum length of generated code")

  args = parser.parse_args()

  # Load model
  print("Loading model...")
  model, processor = load_model(args.model)

  # Generate code
  print(f"Generating D3.js code for {args.image}...")
  code = generate_d3_code(model, processor, args.image, args.output, args.max_length)

  if args.output:
    print(f"Complete HTML with D3.js code saved to {args.output}")
  else:
    print("\nGenerated D3.js code:")
    print("-" * 40)
    print(code)
    print("-" * 40)


if __name__ == "__main__":
  main()
