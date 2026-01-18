import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Конфигурационный файл не найден: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    @property
    def vllm_base_url(self) -> str:
        return self._config['vllm']['base_url']

    @property
    def vllm_model_name(self) -> str:
        return self._config['vllm']['model_name']

    @property
    def vllm_temperature(self) -> float:
        return self._config['vllm']['temperature']

    @property
    def vllm_max_tokens(self) -> int:
        return self._config['vllm']['max_tokens']

    @property
    def vllm_timeout(self) -> int:
        return self._config['vllm']['timeout']

    @property
    def mlflow_experiment_name(self) -> str:
        return self._config['mlflow']['experiment_name']

    @property
    def mlflow_tracking_uri(self) -> str:
        return self._config['mlflow']['tracking_uri']

