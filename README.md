# Fine tuning to generate D3 code from images

## Data

The training data is generated using the `data/graphGenerator.ts` script. This
script generates a set of images and corresponding D3 code. The images and D3
code are stored in the `data/` folder.

### Creating training data

1. Install dependencies

```bash
npm install
```

1. Build the project

```bash
npm run build
```

3. Generate training data

```bash
npm run gen
```

Graphs will be generated in the `data/generated_graphs` folder. D3 code is
generated in the `data/generated_code` folder.

## Training and Inference

For training and inference, you will need to have `python` and `uv` installed.

### Prerequisites

1. Create a virtual environment

```bash
uv venv
source .venv/bin/activate
```

2. Install dependencies

```bash
uv sync
```

### Training

Training is done using the `src/train.py` script with model LoRa weights saved
in the `models/` folder.

```bash
uv run -m src.train
```

### Inference Script

Inference is done using the `src/inference.py` script with model LoRa weights
saved in the `models/` folder.

```bash
uv run -m src.inference --image path/to/chart.png --output generated_code.html
```

#### Inference Options

- `--image`: Path to the input chart image (required)
- `--model`: Path to the fine-tuned model (default:
  "weights/qwen2.5-vl-3b-d3js-finetuned-best")
- `--output`: Output file for the generated D3.js in html format (default:
  "generated_code.html")
- `--max_length`: Maximum length of generated code (default: 2048)
- `--use_original`: Use the original model without fine-tuned weights (flag, no
  value needed)

Example:

```bash
uv run -m src.inference --image data/generated_graphs/graph_1_bar.png --output my_d3_code.html
```

The script will load the fine-tuned model, process the input image, and generate
D3.js code that recreates the chart in the image. The generated code will be
saved wrapped in a `<script>` tag in a html file.

## Application

The application is a simple web application that allows users to upload an image
and generate D3.js code that recreates the chart in the image.

### Running the Application

#### 1. Server

1. Load python environment

```bash
source .venv/bin/activate
```

2. Install dependencies

```bash
uv sync
```

3. Start the server

```bash
uv run -m src.server
```

#### 2. Client

1. Move to client directory

```bash
cd client
```

2. Install dependencies

```bash
npm install
```

3. Start the client

```bash
npm run dev
```

### Accessing the Application

The application can be accessed at `http://localhost:3000`.
