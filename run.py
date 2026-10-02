import subprocess
import sys


def _ld_lines():
    try:
        return subprocess.run(["ldconfig", "-p"], capture_output=True, text=True, timeout=15).stdout.splitlines()
    except Exception:
        return None  # не Linux / нет ldconfig — проверять нечего


def ensure_mediapipe_gl():
    """MediaPipe в headless-окружениях (Debian/Ubuntu slim, Docker) падает с
    OSError: libEGL.so.1 / libGLESv2.so.2 / libGL.so.1: cannot open shared object file.
    Проверяем наличие этих библиотек и при правах root ставим mesa-пакеты автоматически."""
    lines = _ld_lines()
    if lines is None:
        return
    needed = {"libEGL.so.1": "libegl1", "libGLESv2.so.2": "libgles2", "libGL.so.1": "libgl1"}
    missing = [pkg for so, pkg in needed.items() if not any(so in l for l in lines)]
    if not missing:
        return
    cmd = ["apt-get", "update"]
    print(f"[run] Не хватает системных библиотек MediaPipe: {', '.join(missing)}")
    print("[run] Установка:", " ".join(cmd), "&& apt-get install -y " + " ".join(missing))
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        subprocess.run(["apt-get", "install", "-y", *missing], check=True, capture_output=True)
        subprocess.run(["ldconfig"], capture_output=True)
        print("[run] Готово: OpenGL-библиотеки установлены.")
    except Exception as e:
        print(f"[run] НЕ удалось установить автоматически ({e}).\n"
              f"[run] Выполните вручную (нужен root/sudo):\n"
              f"[run]   apt-get update && apt-get install -y {' '.join(missing)}")
        sys.exit(1)


if __name__ == "__main__":
    ensure_mediapipe_gl()
    import uvicorn
    try:
        uvicorn.run("app:app", host="0.0.0.0", port=8000, timeout_graceful_shutdown=3)
    except KeyboardInterrupt:
        print("[stop] сервер остановлен")
