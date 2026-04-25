# -*- coding: utf-8 -*-
"""
KOIB RAG - VK Bot Module

Полноценный VK Long Poll бот для чат-бота по документации КОИБ.
Реализует:
- Хранение сессий in-memory
- Выбор модели КОИБ через кнопки
- Интеграцию с RAG и GigaChat
- Обработку ошибок
"""

import logging
import random
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

import vk_api
from vk_api.longpoll import VkLongPoll, VkEventType
from vk_api.keyboard import VkKeyboard, VkKeyboardColor

from gigachat_client import call_gigachat, GigaChatClient

logger = logging.getLogger(__name__)

# Типы сессий
@dataclass
class UserSession:
    """
    Сессия пользователя.
    
    Атрибуты:
        model: Выбранная модель (koib2010, koib2017a, koib2017b, general)
        history: История последних 3 пар вопрос-ответ
    """
    model: str = "general"
    history: List[Dict[str, str]] = field(default_factory=list)
    
    def add_to_history(self, question: str, answer: str) -> None:
        """Добавить пару вопрос-ответ в историю (максимум 3)."""
        self.history.append({"q": question, "a": answer})
        if len(self.history) > 3:
            self.history.pop(0)
    
    def get_history_text(self) -> str:
        """Получить текст истории для промпта."""
        if not self.history:
            return ""
        
        parts = []
        for pair in self.history[-3:]:
            parts.append(f"Вопрос: {pair['q']}\nОтвет: {pair['a']}")
        
        return "\n\n".join(parts) + "\n\n" if parts else ""
    
    def clear_history(self) -> None:
        """Очистить историю."""
        self.history.clear()


# Константы кнопок
BUTTON_KOIB_2010 = "КОИБ-2010"
BUTTON_KOIB_2017A = "КОИБ-2017А"
BUTTON_KOIB_2017B = "КОИБ-2017Б"
BUTTON_GENERAL = "Общий вопрос"
BUTTON_CHANGE_MODEL = "← Сменить модель"

# Маппинг кнопок на ключи моделей
BUTTON_TO_MODEL = {
    BUTTON_KOIB_2010: "koib2010",
    BUTTON_KOIB_2017A: "koib2017a",
    BUTTON_KOIB_2017B: "koib2017b",
    BUTTON_GENERAL: "general",
}

MODEL_TO_DISPLAY_NAME = {
    "koib2010": "КОИБ-2010",
    "koib2017a": "КОИБ-2017А",
    "koib2017b": "КОИБ-2017Б",
    "general": "общий вопрос",
}


def create_main_keyboard() -> dict:
    """
    Создать основную клавиатуру с выбором модели.
    
    Returns:
        JSON клавиатуры для VK API
    """
    keyboard = VkKeyboard(one_time=False, inline=False)
    
    # Первая строка: КОИБ-2010 и КОИБ-2017А
    keyboard.add_button(BUTTON_KOIB_2010, color=VkKeyboardColor.PRIMARY)
    keyboard.add_button(BUTTON_KOIB_2017A, color=VkKeyboardColor.PRIMARY)
    keyboard.add_line()
    
    # Вторая строка: КОИБ-2017Б и Общий вопрос
    keyboard.add_button(BUTTON_KOIB_2017B, color=VkKeyboardColor.PRIMARY)
    keyboard.add_button(BUTTON_GENERAL, color=VkKeyboardColor.SECONDARY)
    
    return keyboard.get_keyboard()


def create_change_model_keyboard() -> dict:
    """
    Создать клавиатуру с кнопкой "Сменить модель".
    
    Returns:
        JSON клавиатуры для VK API
    """
    keyboard = VkKeyboard(one_time=False, inline=False)
    keyboard.add_button(BUTTON_CHANGE_MODEL, color=VkKeyboardColor.SECONDARY)
    return keyboard.get_keyboard()


class KoibVKBot:
    """
    VK бот для технической поддержки КОИБ.
    
    Атрибуты:
        engine: Query engine для RAG поиска
        system_prompt: Системный промпт
        sessions: Хранилище сессий пользователей
        gigachat_credentials: Учётные данные GigaChat
    """
    
    def __init__(self, engine: Any, system_prompt: str, 
                 gigachat_credentials: Optional[str] = None):
        """
        Инициализировать бота.
        
        Args:
            engine: Query engine с методами ask_with_llm_context
            system_prompt: Системный промпт из prompt.txt
            gigachat_credentials: Учётные данные GigaChat (опционально)
        """
        self.engine = engine
        self.system_prompt = system_prompt
        
        # Получаем credentials из окружения или параметра
        import os
        self.gigachat_credentials = (
            gigachat_credentials or 
            os.environ.get("GIGACHAT_CREDENTIALS") or
            ""
        )
        
        # Хранилище сессий: {user_id: UserSession}
        self.sessions: Dict[int, UserSession] = {}
        
        # VK API
        self.vk_session = None
        self.vk = None
        self.long_poll = None
        
        # Извлекаем приветствие из system_prompt (до разделителя ---)
        self.greeting = self._extract_greeting(system_prompt)
        
        logger.info("✅ VK бот инициализирован")
    
    def _extract_greeting(self, prompt: str) -> str:
        """
        Извлечь приветствие из system prompt (первый абзац до ---).
        
        Args:
            prompt: Полный system prompt
            
        Returns:
            Текст приветствия
        """
        if "---" in prompt:
            greeting = prompt.split("---")[0].strip()
        else:
            # Если нет разделителя, берём первые 2-3 предложения
            lines = prompt.split("\n")
            greeting_parts = []
            for line in lines[:5]:
                if line.strip():
                    greeting_parts.append(line.strip())
            greeting = " ".join(greeting_parts)
        
        return greeting
    
    def _get_or_create_session(self, user_id: int) -> UserSession:
        """
        Получить или создать сессию пользователя.
        
        Args:
            user_id: ID пользователя ВКонтакте
            
        Returns:
            Объект сессии
        """
        if user_id not in self.sessions:
            self.sessions[user_id] = UserSession()
            logger.debug(f"Создана новая сессия для пользователя {user_id}")
        return self.sessions[user_id]
    
    def _reset_session(self, user_id: int) -> UserSession:
        """
        Сбросить сессию пользователя.
        
        Args:
            user_id: ID пользователя
            
        Returns:
            Новая сессия
        """
        self.sessions[user_id] = UserSession()
        logger.debug(f"Сессия пользователя {user_id} сброшена")
        return self.sessions[user_id]
    
    def _send_message(self, user_id: int, text: str, 
                      keyboard: Optional[dict] = None) -> None:
        """
        Отправить сообщение пользователю.
        
        Args:
            user_id: ID пользователя
            text: Текст сообщения
            keyboard: JSON клавиатуры (опционально)
        """
        try:
            params = {
                "user_id": user_id,
                "message": text,
                "random_id": random.randint(1, 2**31 - 1)
            }
            
            if keyboard:
                params["keyboard"] = keyboard
            
            self.vk.messages.send(**params)
            logger.debug(f"Сообщение отправлено пользователю {user_id}")
            
        except Exception as e:
            logger.error(f"Ошибка отправки сообщения: {e}")
    
    def _handle_start(self, user_id: int) -> None:
        """
        Обработать команду старта диалога.
        
        Args:
            user_id: ID пользователя
        """
        session = self._reset_session(user_id)
        
        message = f"{self.greeting}"
        self._send_message(user_id, message, create_main_keyboard())
        logger.info(f"Пользователь {user_id} начал диалог")
    
    def _handle_model_selection(self, user_id: int, button_text: str) -> None:
        """
        Обработать выбор модели.
        
        Args:
            user_id: ID пользователя
            button_text: Текст нажатой кнопки
        """
        session = self._get_or_create_session(user_id)
        
        model_key = BUTTON_TO_MODEL.get(button_text)
        if not model_key:
            logger.warning(f"Неизвестная кнопка: {button_text}")
            return
        
        session.model = model_key
        session.clear_history()
        
        display_name = MODEL_TO_DISPLAY_NAME.get(model_key, model_key)
        message = f"Вы выбрали {display_name}. Задайте ваш вопрос."
        
        self._send_message(user_id, message, create_change_model_keyboard())
        logger.info(f"Пользователь {user_id} выбрал модель {model_key}")
    
    def _answer_question(self, user_id: int, question: str) -> None:
        """
        Обработать вопрос пользователя.
        
        Args:
            user_id: ID пользователя
            question: Текст вопроса
        """
        session = self._get_or_create_session(user_id)
        
        # Проверяем, выбрана ли модель
        if session.model == "general" and not session.history:
            # Пользователь ещё не выбрал модель явно
            # Но для general это допустимо - отвечаем без RAG
            pass
        
        model_key = session.model
        history_text = session.get_history_text()
        
        try:
            if model_key == "general":
                # Общий вопрос - GigaChat без RAG
                prompt = f"{self.system_prompt}\n\n{history_text}Вопрос: {question}"
                answer = call_gigachat(prompt, self.gigachat_credentials)
                
            else:
                # RAG - поиск по конкретной модели
                llm_context = self.engine.ask_with_llm_context(
                    question, 
                    koib_model=model_key
                )
                
                # Проверяем, найден ли контекст
                if "Контекст не найден" in llm_context or not llm_context.strip():
                    answer = "По данной модели информация не найдена. Попробуйте переформулировать вопрос или выберите другую модель."
                else:
                    prompt = (
                        f"{self.system_prompt}\n\n"
                        f"Контекст из документации:\n{llm_context}\n\n"
                        f"{history_text}"
                        f"Вопрос: {question}\n\n"
                        f"Отвечай ТОЛЬКО по документации КОИБ-модели {model_key}. "
                        f"Если информации нет — скажи об этом."
                    )
                    answer = call_gigachat(prompt, self.gigachat_credentials)
            
            # Сохраняем в историю
            session.add_to_history(question, answer)
            
            # Отправляем ответ
            self._send_message(user_id, answer)
            
            logger.info(f"Ответ дан пользователю {user_id} на вопрос: {question[:50]}...")
            
        except Exception as e:
            logger.error(f"Ошибка при обработке вопроса: {e}")
            error_message = "Произошла ошибка при обработке вашего запроса. Попробуйте позже."
            self._send_message(user_id, error_message)
    
    def _handle_message(self, event) -> None:
        """
        Обработать входящее сообщение.
        
        Args:
            event: Событие VK Long Poll
        """
        user_id = event.user_id
        text = event.text.strip() if event.text else ""
        
        logger.debug(f"Получено сообщение от {user_id}: {text[:100]}")
        
        # Команды старта
        start_commands = ["/start", "начать", "привет", "здравствуйте", "старт"]
        if text.lower() in start_commands or text.lower().startswith("/start"):
            self._handle_start(user_id)
            return
        
        # Кнопка смены модели
        if text == BUTTON_CHANGE_MODEL:
            self._handle_start(user_id)
            return
        
        # Проверка нажатия кнопки выбора модели
        if text in BUTTON_TO_MODEL:
            self._handle_model_selection(user_id, text)
            return
        
        # Обычный вопрос
        session = self._get_or_create_session(user_id)
        
        # Если пользователь ещё не выбрал модель (кроме general по умолчанию)
        if session.model == "general" and not session.history:
            # Предлагаем выбрать модель
            self._send_message(
                user_id,
                "Пожалуйста, выберите модель из меню ниже:",
                create_main_keyboard()
            )
            return
        
        # Обрабатываем вопрос
        self._answer_question(user_id, text)
    
    def run(self) -> None:
        """
        Запустить бота (Long Poll).
        
        Метод блокирующий, работает до прерывания.
        """
        import os
        
        vk_token = os.environ.get("VK_GROUP_TOKEN")
        if not vk_token:
            logger.error("❌ VK_GROUP_TOKEN не найден в переменных окружения")
            raise ValueError("VK_GROUP_TOKEN required")
        
        logger.info("🚀 Запуск VK бота...")
        
        # Инициализация VK API
        self.vk_session = vk_api.VkApi(token=vk_token)
        self.vk = self.vk_session.get_api()
        self.long_poll = VkLongPoll(self.vk_session)
        
        logger.info("✅ Бот запущен и ожидает сообщения...")
        
        # Основной цикл Long Poll
        try:
            for event in self.long_poll.listen():
                try:
                    if event.type == VkEventType.MESSAGE_NEW:
                        # Для сообщений от пользователей
                        if event.from_user:
                            self._handle_message(event)
                        
                except Exception as e:
                    logger.error(f"Ошибка обработки события: {e}", exc_info=True)
                    
        except KeyboardInterrupt:
            logger.info("Бот остановлен пользователем")
        except Exception as e:
            logger.error(f"Критическая ошибка Long Poll: {e}", exc_info=True)
            raise


if __name__ == "__main__":
    # Тестовый запуск (требует настроенных переменных окружения)
    logging.basicConfig(level=logging.INFO)
    
    print("Для запуска бота используйте Koib_VK_bot.ipynb в Google Colab")
    print("или установите переменные окружения:")
    print("  - GIGACHAT_CREDENTIALS")
    print("  - VK_GROUP_TOKEN")
