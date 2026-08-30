import io
import os

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

from model import FashionMNISTCNN

MODEL_PATH = os.getenv("MODEL_PATH", "./models/fashion_mnist_cnn.pth")
DEVICE = torch.device("cpu")
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
app = FastAPI(title="Fashion-MNIST Classifier")

model = None
transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ]
)


def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        return
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model = FashionMNISTCNN(num_classes=checkpoint.get("num_classes", 10))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()


load_model()


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        image_bytes = await image.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("L")

        tensor = transform(image)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.to(DEVICE)
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

        result = {CLASS_NAMES[i]: round(probabilities[i].item(), 4) for i in range(10)}
        predicted_class = probabilities.argmax().item()
        return {
            "predicted_class": CLASS_NAMES[predicted_class],
            "probabilities": result,
        }

    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")
