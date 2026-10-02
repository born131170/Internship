"""Запуск PersonaScope: кросс-платформенная подготовка окружения + uvicorn.

Windows/macOS: системные OpenGL-библиотеки НЕ нужны — достаточно pip-зависимостей
(они ставятся командой `pip install -r requirements.txt`, которую этот скрипт
выполнит сам, если чего-то не хватает).

Linux (headless/Docker): при нехватке libEGL/libGLESv2/libGL автоматически ставим
mesa-пакеты через apt/dnf/pacman (если есть root или доступный sudo).
"""
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

SYSTEM = platform.system()          # 'Windows' | 'Linux' | 'Darwin'
IS_WIN = SYSTEM == "Windows"


def _ld_missing():
    """Список недостающих OpenGL-пакетов на Linux (пустой, если всё на месте / не Linux)."""
    if SYSTEM != "Linux":
        return []
    needed = {"libEGL.so.1": "libegl1", "libGLESv2.so.2": "libgles2", "libGL.so.1": "libgl1"}
    try:
        out = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True, timeout=15).stdout
    except Exception:
        return list(needed.values())  # ldconfig недоступен — считаем всё недостающим
    return [pkg for so, pkg in needed.items() if so not in out]


def _root_or_sudo():
    """None — прав нет; [] — мы root; ['sudo'] — sudo работает без пароля."""
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        return []
    if shutil.which("sudo"):
        try:
            if subprocess.run(["sudo", "-n", "true"], capture_output=True, timeout=10).returncode == 0:
                return ["sudo"]
        except Exception:
            pass
    return None


def ensure_mediapipe_gl():
    """Автоустановка OpenGL-библиотек (только Linux). На Windows/macOS не вызывается."""
    missing = _ld_missing()
    if not missing:
        return
    print(f"[run] Не хватает системных библиотек MediaPipe: {', '.join(missing)}")
    prefix = _root_or_sudo()
    if prefix is None:
        print("[run] Нет прав root и sudo не доступен без пароля — установите вручную:\n"
              f"[run]   sudo apt-get update && sudo apt-get install -y {' '.join(missing)}")
        sys.exit(1)
    if shutil.which("apt-get"):
        cmds = [["apt-get", "update"], ["apt-get", "install", "-y", *missing], ["ldconfig"]]
    elif shutil.which("dnf"):
        cmds = [["dnf", "install", "-y", "mesa-libEGL", "mesa-libGLES", "mesa-libGL"], ["ldconfig"]]
    elif shutil.which("pacman"):
        cmds = [["pacman", "-S", "--noconfirm", "mesa", "libglvnd"]]
    else:
        print("[run] Неизвестный пакетный менеджер. Установите libegl1/libgles2/libgl1 вручную.")
        sys.exit(1)
    for c in cmds:
        try:
            subprocess.run(prefix + c, check=True, capture_output=True)
        except Exception as e:
            print(f"[run] Ошибка команды '{' '.join(c)}': {e}\n"
                  f"[run] Выполните вручную: sudo {' '.join(c)}")
            sys.exit(1)
    print("[run] Готово: OpenGL-библиотеки установлены.")


def ensure_python_deps():
    """Ставит pip-зависимости из requirements.txt, если какого-то модуля нет."""
    need_check = {"mediapipe": "mediapipe", "cv2": "opencv-python", "fastapi": "fastapi",
                  "uvicorn": "uvicorn", "numpy": "numpy", "requests": "requests"}
    missing = [pkg for mod, pkg in need_check.items() if importlib.util.find_spec(mod) is None]
    if not missing:
        return
    print(f"[run] Не хватает python-пакетов: {', '.join(missing)}. Устанавливаю из requirements.txt...")
    req = Path(__file__).parent / "requirements.txt"
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req)])
    if r.returncode != 0:
        print("[run] pip install не удался. Выполните вручную:\n"
              f"[run]   {sys.executable} -m pip install -r requirements.txt")
        sys.exit(1)
    print("[run] Python-зависимости установлены.")


if __name__ == "__main__":
    ensure_python_deps()
    if not IS_WIN:
        ensure_mediapipe_gl()
    import uvicorn
    port = int(os.environ.get("PORT", 8001 if IS_WIN else 8000))
    print(f"[run] PersonaScope: http://127.0.0.1:{port}  (Ctrl+C — остановка сервера)")
    try:
        uvicorn.run("app:app", host="0.0.0.0", port=port, timeout_graceful_shutdown=3)
    except OSError as e:
        print(f"[run] Не удалось запустить сервер на порту {port}: {e}\n"
              f"[run] Задайте другой порт: set PORT=8080 (Windows) / export PORT=8080 (Linux)")
        sys.exit(1)
    except KeyboardInterrupt:
        print("[stop] Сервер остановлен.")
