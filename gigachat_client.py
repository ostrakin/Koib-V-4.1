# -*- coding: utf-8 -*-
"""
KOIB RAG - GigaChat Client

Клиент для работы с GigaChat API.
Выполняет OAuth2 аутентификацию и отправку запросов к LLM.
"""

import logging
import time
from typing import Optional
import requests

logger = logging.getLogger(__name__)

# Константы GigaChat API
GIGACHAT_TOKEN_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
GIGACHAT_CHAT_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
GIGACHAT_SCOPE = "GIGACHAT_API_PERS"
GIGACHAT_MODEL = "GigaChat"
GIGACHAT_TEMPERATURE = 0.3
GIGACHAT_MAX_TOKENS = 1024
GIGACHAT_TIMEOUT = 30  # секунд


class GigaChatClient:
    """
    Клиент для работы с GigaChat API.
    
    Атрибуты:
        credentials: Базовые учётные данные (client_id:client_secret в base64)
        access_token: Текущий access token
        token_expires_at: Время истечения токена
    """
    
    def __init__(self, credentials: str):
        """
        Инициализировать клиент GigaChat.
        
        Args:
            credentials: Строка авторизации (client_id:client_secret в base64)
        """
        self.credentials = credentials
        self.access_token: Optional[str] = None
        self.token_expires_at: float = 0
    
    def _get_token(self) -> str:
        """
        Получить или обновить access token через OAuth2.
        
        Returns:
            Access token
            
        Raises:
            RuntimeError: Если не удалось получить токен
        """
        # Проверяем, действителен ли текущий токен
        if self.access_token and time.time() < self.token_expires_at - 60:
            # Токен действителен ещё минимум 60 секунд
            return self.access_token
        
        logger.info("Получение нового токена GigaChat...")
        
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": f"Basic {self.credentials}",
            "RqUID": "00000000-0000-0000-0000-000000000000"
        }
        
        payload = {
            "scope": GIGACHAT_SCOPE
        }
        
        try:
            response = requests.post(
                GIGACHAT_TOKEN_URL,
                headers=headers,
                data=payload,
                timeout=10,
                verify=False  # Требуется для самоподписанных сертификатов Сбера
            )
            
            if response.status_code != 200:
                logger.error(f"Ошибка получения токена: {response.status_code} - {response.text}")
                raise RuntimeError(f"GigaChat OAuth error: {response.status_code}")
            
            data = response.json()
            self.access_token = data.get("access_token")
            
            # Время жизни токена (в секундах)
            expires_in = data.get("expires_in", 1800)
            self.token_expires_at = time.time() + expires_in
            
            logger.info(f"✅ Токен получен, истекает через {expires_in} сек")
            return self.access_token
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка сети при получении токена: {e}")
            raise RuntimeError(f"GigaChat connection error: {e}")
    
    def chat(self, prompt: str, temperature: float = GIGACHAT_TEMPERATURE,
             max_tokens: int = GIGACHAT_MAX_TOKENS) -> str:
        """
        Отправить запрос к GigaChat и получить ответ.
        
        Args:
            prompt: Текст запроса
            temperature: Температура генерации (0.0-1.0)
            max_tokens: Максимальное количество токенов в ответе
            
        Returns:
            Текст ответа от GigaChat
        """
        # Получаем токен
        token = self._get_token()
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        
        payload = {
            "model": GIGACHAT_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        try:
            logger.debug(f"Отправка запроса к GigaChat ({len(prompt)} символов)...")
            
            response = requests.post(
                GIGACHAT_CHAT_URL,
                headers=headers,
                json=payload,
                timeout=GIGACHAT_TIMEOUT
            )
            
            if response.status_code != 200:
                logger.error(f"Ошибка GigaChat API: {response.status_code} - {response.text}")
                
                # Пробуем обновить токен и повторить запрос
                if response.status_code == 401:
                    logger.info("Токен истёк, получаем новый...")
                    self.access_token = None
                    token = self._get_token()
                    headers["Authorization"] = f"Bearer {token}"
                    
                    response = requests.post(
                        GIGACHAT_CHAT_URL,
                        headers=headers,
                        json=payload,
                        timeout=GIGACHAT_TIMEOUT
                    )
                    
                    if response.status_code != 200:
                        return "Сервис временно недоступен. Попробуйте позже."
                
                else:
                    return "Сервис временно недоступен. Попробуйте позже."
            
            data = response.json()
            
            # Извлекаем ответ из структуры GigaChat
            choices = data.get("choices", [])
            if not choices:
                logger.warning("GigaChat вернул пустой список choices")
                return "Не удалось получить ответ от сервиса."
            
            answer = choices[0].get("message", {}).get("content", "")
            
            if not answer:
                logger.warning("GigaChat вернул пустой ответ")
                return "Не удалось получить ответ от сервиса."
            
            logger.debug(f"✅ Получен ответ от GigaChat ({len(answer)} символов)")
            return answer
            
        except requests.exceptions.Timeout:
            logger.error("Таймаут запроса к GigaChat")
            return "Сервис временно недоступен (таймаут). Попробуйте позже."
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка сети при запросе к GigaChat: {e}")
            return "Сервис временно недоступен. Попробуйте позже."
            
        except Exception as e:
            logger.error(f"Неожиданная ошибка при работе с GigaChat: {e}")
            return "Произошла ошибка при обработке запроса."


def call_gigachat(prompt: str, credentials: str, 
                  temperature: float = GIGACHAT_TEMPERATURE,
                  max_tokens: int = GIGACHAT_MAX_TOKENS) -> str:
    """
    Удобная функция для вызова GigaChat.
    
    Args:
        prompt: Текст запроса
        credentials: Учётные данные (client_id:client_secret в base64)
        temperature: Температура генерации
        max_tokens: Максимальное количество токенов
        
    Returns:
        Текст ответа
    """
    client = GigaChatClient(credentials)
    return client.chat(prompt, temperature, max_tokens)


if __name__ == "__main__":
    # Тестовый запуск
    import os
    
    credentials = os.environ.get("GIGACHAT_CREDENTIALS")
    if not credentials:
        print("❌ Установите переменную окружения GIGACHAT_CREDENTIALS")
    else:
        print("Тестовый запрос к GigaChat...")
        response = call_gigachat("Привет! Как дела?", credentials)
        print(f"Ответ: {response}")
