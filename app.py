import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field


app = FastAPI(
    title="Profesora IA EIEB",
    version="0.1.0",
)

# Para el MVP permitimos llamadas desde Tiendup.
# Luego restringiremos esto al dominio exacto de tu plataforma.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError(
        "Falta configurar la variable OPENAI_API_KEY en Render."
    )

client = OpenAI(api_key=api_key)


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Pregunta realizada por la alumna.",
    )


class ChatResponse(BaseModel):
    answer: str


SYSTEM_PROMPT = """
Sos la Profesora IA oficial de EIEB, especializada inicialmente
en Reflexología Podal.

Tu función es ayudar a las alumnas que ya compraron el curso
a comprender sus contenidos.

Reglas principales:

- Respondé en español claro, amable, profesional y educativo.
- Explicá como una profesora paciente.
- Priorizá la comprensión por encima de la memorización.
- Cuando sea útil, explicá paso a paso y utilizá ejemplos.
- No inventes información.
- Si no tenés información suficiente, decilo claramente.
- No diagnostiques enfermedades.
- No reemplaces la consulta con un profesional de la salud.
- No afirmes que la reflexología cura enfermedades.
- Diferenciá la orientación educativa del asesoramiento médico.
- Respondé solamente consultas pedagógicas relacionadas
  con Reflexología, estética, bienestar y el aprendizaje del curso.
- Si preguntan algo totalmente ajeno, indicá amablemente que
  tu función es acompañar el aprendizaje dentro de EIEB.
- Evitá respuestas innecesariamente largas.
"""


@app.get("/")
def home():
    return {
        "message": "Profesora IA de EIEB funcionando",
        "version": "0.1.0",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "openai_configured": bool(api_key),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(data: ChatRequest):
    question = data.message.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="La pregunta no puede estar vacía.",
        )

    try:
        response = client.responses.create(
            model="gpt-5.5",
            instructions=SYSTEM_PROMPT,
            input=question,
        )

        answer = response.output_text.strip()

        if not answer:
            raise HTTPException(
                status_code=502,
                detail="OpenAI no devolvió una respuesta.",
            )

        return ChatResponse(answer=answer)

    except HTTPException:
        raise

    except Exception as error:
        print(f"Error de OpenAI: {error}")

        raise HTTPException(
            status_code=500,
            detail="No fue posible generar la respuesta.",
        )
