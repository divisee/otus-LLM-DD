.PHONY: help install check test experiment genai mlflow clean

help:
	@echo "Доступные команды:"
	@echo "  make install    - Установить зависимости"
	@echo "  make check      - Проверить подключение к vLLM"
	@echo "  make test       - Тестировать API (requests/httpx/openai)"
	@echo "  make experiment - Запустить эксперимент"
	@echo "  make genai      - Эксперимент с make_genai_metric"
	@echo "  make mlflow     - Запустить MLflow UI"
	@echo "  make clean      - Очистить временные файлы"

install:
	@echo "Установка зависимостей..."
	pip install openai requests httpx pyyaml pandas mlflow
	@echo "✅ Установка завершена"

check:
	@echo "Проверка vLLM сервера..."
	python check_vllm.py

test:
	@echo "Тестирование vLLM API..."
	python test_vllm_api.py

experiment:
	@echo "Запуск эксперимента..."
	python run_experiment.py

genai:
	@echo "Запуск эксперимента с make_genai_metric..."
	python run_genai_metric_experiment.py

mlflow:
	@echo "Запуск MLflow UI на http://localhost:5000"
	@echo "Tracking URI: sqlite:///mlflow.db"
	cd $(shell pwd) && mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

clean:
	@echo "Очистка временных файлов..."
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf *.pyc
	@echo "✅ Очистка завершена"
