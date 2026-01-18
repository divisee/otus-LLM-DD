import requests
import httpx
from openai import OpenAI
from typing import Dict, List, Optional
from config_loader import Config


class VLLMClient:
    def __init__(self, config: Config):
        self.config = config
        self.base_url = config.vllm_base_url
        self.model_name = config.vllm_model_name
        self.temperature = config.vllm_temperature
        self.max_tokens = config.vllm_max_tokens
        self.timeout = config.vllm_timeout

        self.openai_client = OpenAI(
            base_url=self.base_url,
            api_key="EMPTY"
        )

    def chat_with_requests(self, messages: List[Dict[str, str]],
                          temperature: Optional[float] = None,
                          max_tokens: Optional[int] = None) -> str:
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens
        }

        response = requests.post(
            url,
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()

        result = response.json()
        return result['choices'][0]['message']['content']

    def chat_with_httpx(self, messages: List[Dict[str, str]],
                       temperature: Optional[float] = None,
                       max_tokens: Optional[int] = None) -> str:
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens
        }

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()

            result = response.json()
            return result['choices'][0]['message']['content']

    def chat_with_openai(self, messages: List[Dict[str, str]],
                        temperature: Optional[float] = None,
                        max_tokens: Optional[int] = None) -> str:
        response = self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens
        )

        return response.choices[0].message.content

    def simple_query(self, question: str, method: str = "openai") -> str:
        messages = [
            {"role": "user", "content": question}
        ]

        if method == "requests":
            return self.chat_with_requests(messages)
        elif method == "httpx":
            return self.chat_with_httpx(messages)
        elif method == "openai":
            return self.chat_with_openai(messages)
        else:
            raise ValueError(f"Неизвестный метод: {method}")
