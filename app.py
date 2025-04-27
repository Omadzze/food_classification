import io

import torch.jit
from PIL.Image import Image
from fastapi import FastAPI, UploadFile, File

from torchvision.transforms import transforms

app = FastAPI()
model = torch.jit.torch.jit.load("model.pt")
model.eval()


# preprocess image
preprocess_image = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    laoded_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    x = preprocess_image(laoded_image).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        prediction = logits.argmax(-1).item()
    return {"predicted_class": prediction}