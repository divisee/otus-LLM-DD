#!/bin/bash

echo "Запуск MLflow UI из папки hw-3"
echo "Tracking URI: sqlite:///mlflow.db"
echo ""

cd "$(dirname "$0")"
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
