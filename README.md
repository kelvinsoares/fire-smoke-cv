# FireSmokeVision

Sistema de detecção de fogo e fumaça em imagens utilizando Visão Computacional e Deep Learning com YOLOv8.

## Objetivo

Desenvolver uma solução completa de visão computacional para detecção de eventos de risco (fogo e fumaça) a partir de imagens, simulando aplicações reais de monitoramento por câmeras.

## Tecnologias utilizadas

- Python
- YOLOv8
- OpenCV
- PyTorch
- FastAPI
- Docker
- Pytest
- Ruff
- Google Colab (GPU)

## Dataset

Dataset público de detecção de fogo e fumaça (Kaggle), contendo imagens anotadas para treinamento de modelos YOLO.

## Modelo

- Arquitetura: YOLOv8n
- Tarefa: Object Detection
- Classes:
  - Smoke
  - Fire

## Treinamento

Treinamento realizado no Google Colab com GPU.

Parâmetros principais:

- Epochs: 30
- Image size: 640
- Batch size: 8

## Resultados

Resultados obtidos após treinamento:

![Results](results/metrics/results.png)

![Confusion Matrix](results/metrics/confusion_matrix.png)

## Exemplos de detecção

Algumas predições do modelo:

![Example](results/examples/1.jpg)
![Example](results/examples/2.jpg)

##  Como executar

```bash
uv venv
source .venv/bin/activate

uv add ultralytics opencv-python

uv run yolo detect train \
  model=yolov8n.pt \
  data=configs/fire_smoke.yaml

```

## API com FastAPI

A API permite enviar uma imagem e receber:

✔ JSON com detecções  
✔ Imagem com bounding boxes  

### Rodar localmente

```bash
uv run uvicorn app.main:app --reload
http://127.0.0.1:8000/docs
```
Exemplo (JSON):
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@image.jpg"
```
## Testes automatizados

O projeto possui testes automatizados para validar as rotas principais da API.

```bash
uv run pytest -v
uv run ruff check .
```

## Docker

### Build

```bash
docker build -t fire-smoke-api .
docker run -p 8000:8000 fire-smoke-api

# Acesse:
http://localhost:8000/docs/

## Estrutura do projeto

```bash
fire-smoke-cv/
│
├── app/                # API FastAPI
├── scripts/            # Scripts auxiliares
├── configs/            # Configuração do dataset
├── tests/              # Testes automatizados
├── results/
│   ├── examples/
│   ├── metrics/
├── models/
├── Dockerfile
├── README.md

```

## Melhorias futuras

- Deploy em nuvem (AWS/GCP)
- Streaming de vídeo em tempo real
- Pipeline CI/CD

## 👨‍💻Autor
Kelvin Soares
linkedin.com/in/eukelvinsoares
