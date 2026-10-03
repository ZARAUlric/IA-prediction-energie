from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from src.data.loader import load_dataset, get_target, load_feature_list, load_scaler

app = FastAPI(
    title="API Prédiction Énergie",
    description="API pour le projet de prédiction de consommation énergétique",
    version="0.1.0"
)

# CORS pour Gradio
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DatasetInfo(BaseModel):
    name: str
    target: str
    n_features: int
    train_rows: int
    val_rows: int
    test_rows: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str


class DataSample(BaseModel):
    dataset: str
    split: str
    n_rows: int = 10


class PredictionRequest(BaseModel):
    dataset: str
    features: Dict[str, float]


class ModelLoadRequest(BaseModel):
    model_type: str  # "xgboost", "random_forest", "prophet"
    dataset: str
    model_path: Optional[str] = None


@app.get("/")
async def root():
    return {
        "message": "API Prédiction Énergie",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "datasets": "/datasets",
            "dataset_info": "/datasets/{name}/info",
            "data_sample": "/data/sample",
            "load_model": "/model/load",
            "predict": "/predict",
            "model_info": "/model/info"
        }
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/datasets")
async def list_datasets():
    """Liste les datasets disponibles"""
    return {"datasets": ["household", "appliances"]}


@app.get("/datasets/{name}/info", response_model=DatasetInfo)
async def get_dataset_info(name: str):
    """Informations sur un dataset"""
    if name not in ["household", "appliances"]:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        train = load_dataset(name, "train")
        val = load_dataset(name, "val")
        test = load_dataset(name, "test")
        target = get_target(name)
        features = load_feature_list(name)
        
        return DatasetInfo(
            name=name,
            target=target,
            n_features=len(features),
            train_rows=len(train),
            val_rows=len(val),
            test_rows=len(test),
            train_start=str(train.index.min()),
            train_end=str(train.index.max()),
            test_start=str(test.index.min()),
            test_end=str(test.index.max())
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/data/sample")
async def get_data_sample(request: DataSample):
    """Échantillon de données"""
    if request.dataset not in ["household", "appliances"]:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if request.split not in ["train", "val", "test"]:
        raise HTTPException(status_code=400, detail="Invalid split")
    
    try:
        df = load_dataset(request.dataset, request.split)
        sample = df.head(request.n_rows)
        
        # Convertir en dict pour JSON
        data = {
            "index": [str(idx) for idx in sample.index],
            "columns": list(sample.columns),
            "data": sample.values.tolist(),
            "shape": sample.shape
        }
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/datasets/{name}/features")
async def get_features(name: str):
    """Liste des features d'un dataset"""
    if name not in ["household", "appliances"]:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        features = load_feature_list(name)
        target = get_target(name)
        return {
            "dataset": name,
            "target": target,
            "n_features": len(features),
            "features": features
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Stockage global du modèle chargé
loaded_model = None
model_metadata = {}


@app.post("/model/load")
async def load_model(request: ModelLoadRequest):
    """Charge un modèle en mémoire"""
    global loaded_model, model_metadata
    
    if request.dataset not in ["household", "appliances"]:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        # Chemin par défaut vers le modèle
        if request.model_path is None:
            model_dir = Path("models")
            model_path = model_dir / f"{request.model_type}_{request.dataset}.joblib"
        else:
            model_path = Path(request.model_path)
        
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model file not found: {model_path}")
        
        # Charger le modèle
        loaded_model = joblib.load(model_path)
        
        # Métadonnées
        model_metadata = {
            "model_type": request.model_type,
            "dataset": request.dataset,
            "model_path": str(model_path),
            "features": load_feature_list(request.dataset),
            "target": get_target(request.dataset)
        }
        
        return {
            "status": "loaded",
            "model_type": request.model_type,
            "dataset": request.dataset,
            "model_path": str(model_path)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
async def get_model_info():
    """Informations sur le modèle chargé"""
    if loaded_model is None:
        raise HTTPException(status_code=404, detail="No model loaded")
    
    return {
        "status": "loaded",
        **model_metadata
    }


@app.post("/predict")
async def predict(request: PredictionRequest):
    """Fait une prédiction avec le modèle chargé"""
    global loaded_model
    
    if loaded_model is None:
        raise HTTPException(status_code=404, detail="No model loaded. Use /model/load first")
    
    if request.dataset != model_metadata["dataset"]:
        raise HTTPException(status_code=400, detail="Dataset mismatch")
    
    try:
        # Préparer les features
        features = model_metadata["features"]
        feature_values = []
        
        for feat in features:
            if feat not in request.features:
                raise HTTPException(status_code=400, detail=f"Missing feature: {feat}")
            feature_values.append(request.features[feat])
        
        # Créer le DataFrame
        X = pd.DataFrame([feature_values], columns=features)
        
        # Prédiction
        prediction = loaded_model.predict(X)[0]
        
        return {
            "prediction": float(prediction),
            "dataset": request.dataset,
            "model_type": model_metadata["model_type"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
