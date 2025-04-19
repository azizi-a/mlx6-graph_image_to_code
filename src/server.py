import fastapi
import uvicorn
import io
import PIL
import sys

import src.constants
import src.inference


args = sys.argv[1:]
is_prod = [x for x in args if x == "--prod"]
is_prod = bool(len(is_prod))

app = fastapi.FastAPI(title="D3.js Generator", description="Generate D3.js code for graphs from images")


@app.get("/health")
async def health_check():
  return {"status": "ok"}


@app.post("/generate")
async def generate_d3_code(file: fastapi.UploadFile = fastapi.File(...)):
  # Read the uploaded file into memory
  image_bytes = await file.read()
  image = PIL.Image.open(io.BytesIO(image_bytes))

  # Convert to RGB if needed
  if image.mode != "RGB":
    image = image.convert("RGB")

  # Generate code
  code = await src.inference.load_model_and_generate_d3_code(src.constants.MODEL_PATH, image)
  return {"code": code}


#
#
#
def start_server(host="0.0.0.0", port=8000, debug=False):
  """
  Start the FastAPI server with uvicorn
  """
  uvicorn.run("src.server:app", host=host, port=port, reload=debug)


if __name__ == "__main__":
  if is_prod:
    start_server()
  else:
    start_server(debug=True)
