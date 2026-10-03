FROM python:3.13-slim

WORKDIR /app

# Copier les fichiers de dépendances
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY src/ ./src/
COPY configs/ ./configs/
COPY data/ ./data/
COPY models/ ./models/

# Exposer les ports
EXPOSE 8000 7860

# Commande par défaut (API)
CMD ["python", "-m", "src.api.main"]
