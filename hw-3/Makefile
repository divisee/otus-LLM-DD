.PHONY: help install check test experiment advanced mlflow clean

help:
	@echo "Доступные команды:"
	@echo "  make install    - Установить зависимости"
	@echo "  make check      - Проверить подключение к vLLM"
	@echo "  make test       - Тестировать API (requests/httpx/openai)"
	@echo "  make experiment - Запустить основной эксперимент"
	@echo "  make advanced   - Запустить расширенный эксперимент"
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
	@echo "Запуск основного эксперимента..."
	python run_experiment.py

advanced:
	@echo "Запуск расширенного эксперимента..."
	python run_advanced_experiment.py

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
