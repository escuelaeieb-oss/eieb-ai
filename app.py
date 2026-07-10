from fastapi import FastAPI

app = FastAPI(title="EIEB AI")

@app.get("/")
def home():
    return {
        "mensaje": "Profesor IA de EIEB funcionando"
    }

@app.get("/health")
def health():
    return {
        "status": "ok"
    }
