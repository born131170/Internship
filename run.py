import uvicorn
if __name__=="__main__":
    try:
        uvicorn.run("app:app",host="0.0.0.0",port=8000,timeout_graceful_shutdown=3)
    except KeyboardInterrupt:
        print("[stop] сервер остановлен")
