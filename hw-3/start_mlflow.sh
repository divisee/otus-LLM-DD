#!/bin/bash

echo "Запуск MLflow UI из папки hw-3"
echo "Tracking URI: $(pwd)/mlruns"
echo ""

cd "$(dirname "$0")"
mlflow ui --backend-store-uri ./mlruns --port 5000
