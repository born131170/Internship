# PersonaScope — запуск на Windows

Анализ личности по видео: MediaPipe (локальный движок) + LLM (внешний API).

## Запуск (после получения свежего кода)

```bat
cd C:\PersonaScope
python run.py
```

`run.py` сам:
- проверит Python-зависимости и при нехватке выполнит `pip install -r requirements.txt`;
- на Windows **не** трогает OpenGL-библиотеки (они не нужны);
- поднимет сервер и напишет адрес, обычно http://127.0.0.1:8001

Ручная подготовка окружения (если автоустановка не сработала):

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python run.py
```

Требования: Windows x64, Python 3.9–3.12 (mediapipe не собирается на 3.13 —
если у вас 3.13, поставьте рядом 3.11/3.12 и запускайте через него),
Microsoft Visual C++ Redistributable 2015-2022 (x64).


## Настройка LLM

Ключ внешнего API задаётся в интерфейсе (вкладка настроек) или в файле `.env.local`
(не коммитить!). Локальный движок MediaPipe работает без интернета.
