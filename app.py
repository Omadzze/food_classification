import io
import json

import torch.jit
from PIL.Image import Image
from fastapi import FastAPI, UploadFile, File

from torchvision.transforms import transforms

app = FastAPI()
# Load model
model = torch.jit.torch.jit.load("model.pt")
model.eval()

# dumb json file with labels
with open("class_names.json", "r") as f:
    class_names = json.load(f)


# preprocess image
preprocess_image = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # loading image
    img_bytes = await file.read()
    loaded_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    x = preprocess_image(loaded_image).unsqueeze(0)
    # Model inference
    with torch.no_grad():
        logits = model(x)
        prediction = logits.argmax(-1).item()

    return {"predicted_class": prediction,
            "predicted_label": class_names[prediction]}