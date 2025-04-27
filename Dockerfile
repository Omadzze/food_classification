FROM python:3.9-slim
LABEL authors="omad"

RUN pip install torch torchvision fastapi uvicorn pillow

WORKDIR /app
COPY model.pt app.py ./

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]