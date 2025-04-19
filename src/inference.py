import torch
import PIL
import argparse
import transformers
import peft
import os


#
#
#
def load_model(model_path, use_original=False):
  """
  Load the fine-tuned model and processor

  Args:
    model_path: Path to the fine-tuned model
    use_original: If True, use the original model without fine-tuned weights
  """
  # Load processor from the base model, not the fine-tuned path
  processor = transformers.AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
  processor.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

  # Load base model
  base_model = transformers.AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", device_map="auto", torch_dtype=torch.float16
  )
  base_model.resize_token_embeddings(len(processor.tokenizer))

  if use_original:
    # Return the original model without fine-tuned weights
    model = base_model
  else:
    # Check if model path exists
    if not os.path.exists(model_path):
      raise FileNotFoundError(f"Model path {model_path} does not exist. Please run training first.")

    # Load fine-tuned model
    model = peft.PeftModel.from_pretrained(base_model, model_path)

  model.eval()
  return model, processor


#
#
#
def extract_code(code):
  """
  Extract just the code part (remove any assistant prefixes)
  """

  if "assistant" in code.lower():
    code = code.split("assistant")[-1].strip()
    if code.startswith(":"):
      code = code[1:].strip()

  if "```javascript" in code:
    code = code.split("```javascript")[1]
    code = code.split("```")[0]

  return code


#
#
#
def generate_d3_code(model, processor, image=None, image_path=None, max_length=1024):
  """
  Generate D3.js code for a graph image
  """

  if image is None and image_path is None:
    raise ValueError("Either image or image_path must be provided")

  if image is None:
    # Load and process the image
    image = PIL.Image.open(image_path).convert("RGB")

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

  generation_config = {
    "max_length": max_length,
    "do_sample": True,
    "num_beams": 1,
    "temperature": 0.1,
  }

  # Generate code
  with torch.no_grad():
    outputs = model.generate(**inputs, **generation_config)

  # Decode the generated text
  generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

  code = extract_code(generated_text)

  return code


#
#
#
def load_model_and_generate_d3_code(model_path, image=None, image_path=None, max_length=1024):
  """
  Load the model and generate D3.js code for a graph image
  """

  # Load model
  model, processor = load_model(model_path)

  # Generate code
  code = generate_d3_code(model, processor, image, image_path, max_length)
  return code


#
#
#
def save_html(code, output_path):
  # Create a complete HTML page with the generated D3.js code
  html = f"""<!DOCTYPE html>
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
  # If output doesn't end with .html, add .html extension
  if not output_path.endswith(".html"):
    output_path = f"{output_path}.html"

  with open(output_path, "w") as f:
    f.write(html)
  print(f"Complete HTML with D3.js code saved to {output_path}")


#
#
#
def main():
  parser = argparse.ArgumentParser(description="Generate D3.js code from graph images")
  parser.add_argument("--image", type=str, required=True, help="Path to the graph image")
  parser.add_argument(
    "--model", type=str, default="weights/qwen2.5-vl-3b-d3js-finetuned-best", help="Path to the fine-tuned model"
  )
  parser.add_argument(
    "--output",
    type=str,
    default="./inference_results/graph.html",
    help="Path to save the generated HTML with D3.js code (optional)",
  )
  parser.add_argument("--max_length", type=int, default=2048, help="Maximum length of generated code")
  parser.add_argument("--use_original", action="store_true", help="Use the original model without fine-tuned weights")

  args = parser.parse_args()

  # Load model
  print("Loading model...")
  model, processor = load_model(args.model, args.use_original)

  # Generate code
  print(f"Generating D3.js code for {args.image}...")
  code = generate_d3_code(
    model,
    processor,
    image_path=args.image,
    max_length=args.max_length,
  )

  # Save HTML
  if args.output:
    save_html(code, args.output)

  if args.output:
    print(f"Complete HTML with D3.js code saved to {args.output}")
  else:
    print("\nGenerated D3.js code:")
    print("-" * 40)
    print(code)
    print("-" * 40)


if __name__ == "__main__":
  main()
