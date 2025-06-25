# Food Classification with PyTorch


This project explores various approaches to classify images of [101 food categories](https://docs.pytorch.org/vision/0.18/generated/torchvision.datasets.Food101.html) dataset. Key activities include:

- Loading and augmenting data with PyTorch’s DataLoader
- Benchmarking pretrained models like MobileNet v3 and other models
- Implementing a custom CNN model
- Tracking experiments with Weights & Biases
- Serving predictions via FastAPI app
- Creating Docker container for inference

## Quick Start

1. **Clone the repository**
   ```bash
    git clone https://github.com/Omadzze/food_classification.git
    cd food_classification
    ```
2. **Create and activate a virtual environment**
   ```bash
   conda env create -f environment.yml
   conda activate food_classification
    ```
3. **Prepare the Food-101 data**
   ```bash
   Download from: https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/
   unzip food-101.zip -d data/
    ```
4. **Prepare the Food-101 data**
   ```bash
   python experiments.py --model mobilenet_v3_large --epochs 10
   ```
   
5. **Serve the trained model via FastAP**
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
    ```
    
6. **Build and run Docker container for inference**
   ```bash
   docker build -t foodclf:latest 
   docker run -p 8000:8000 foodclf:latest
    ```
## Directory Structure
```bash
food_classification/
├── data/            # Food-101 images
├── src/             # Data and training scripts
├── model/           # Custom CNN code
├── experiments.py   # Pretrained model runs
├── app.py           # FastAPI inference service
├── Dockerfile       # Container for serving
└── README.md
```

---

## Results Snapshot

| Model                | Accuracy. |
| -------------------- |-----------|
| Pretrained MobileNet | \~81%     |
| Custom CNN           | \~65%     |



