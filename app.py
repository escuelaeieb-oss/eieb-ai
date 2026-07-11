import logging
import os
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel, Field


# ---------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eieb-ai")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.5")
OPENAI_VECTOR_STORE_ID = os.getenv("OPENAI_VECTOR_STORE_ID")

client = (
    OpenAI(api_key=OPENAI_API_KEY)
    if OPENAI_API_KEY
    else None
)


# ---------------------------------------------------------
# APLICACIÓN
# ---------------------------------------------------------

app = FastAPI(
    title="Profesora IA EIEB",
    description="Asistente pedagógico de EIEB.",
    version="0.5.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static",
)


# ---------------------------------------------------------
# MODELOS
# ---------------------------------------------------------

class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(
        ...,
        min_length=1,
        max_length=8000,
    )


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
    )

    history: list[HistoryMessage] = Field(
        default_factory=list,
        description="Últimos mensajes de la conversación.",
    )


class ChatResponse(BaseModel):
    answer: str
    model: str


# ---------------------------------------------------------
# PROMPT
# ---------------------------------------------------------

SYSTEM_PROMPT = """
Sos la Profesora IA oficial de la Escuela Iberoamericana de
Estética y Bienestar, EIEB.

Estás especializada en Reflexología Podal y acompañás a alumnas
que ya compraron el curso.

ORDEN DE PRIORIDAD

1. Interpretá primero la pregunta usando la conversación reciente.
2. Si la alumna dice "la tercera", "eso", "esa parte",
   "lo anterior" o algo parecido, buscá el referente en los
   mensajes anteriores de la conversación.
3. Solo después consultá los documentos del curso mediante
   File Search para ampliar o verificar la respuesta.
4. No interpretes una referencia conversacional como un número
   de paso, página o apartado del manual salvo que la alumna
   lo indique expresamente.

EJEMPLO

Profesora:
1. Lesiones agudas.
2. Trastornos circulatorios.
3. Embarazo.

Alumna:
"Explicame la tercera más fácil."

Interpretación correcta:
La alumna se refiere al embarazo, porque fue el tercer elemento
de la respuesta anterior.

Interpretación incorrecta:
Buscar el paso 3 o el punto 3 del manual.

DOCUMENTOS DEL CURSO

- Consultá primero los documentos oficiales cuando necesites
  información sobre Reflexología Podal.
- Si encontrás la respuesta, basate principalmente en ellos.
- Si no hay información suficiente, decilo claramente.
- No inventes páginas, clases, módulos ni referencias.
- Si complementás con conocimiento general, aclaralo.

FORMA DE ENSEÑAR

- Respondé siempre en español.
- Usá un tono amable, profesional, paciente y cercano.
- Explicá con palabras sencillas.
- Definí los términos técnicos.
- Dividí las explicaciones complejas en pasos.
- Usá ejemplos cuando ayuden.
- No repitas exactamente una explicación si la alumna dice
  que no entendió.
- Evitá respuestas innecesariamente extensas.
- No llenes las respuestas de emojis.

SEGURIDAD

- No diagnostiques enfermedades.
- No indiques tratamientos médicos.
- No afirmes que la reflexología cura enfermedades.
- No prometas resultados terapéuticos.
- No reemplaces a profesionales de la salud.
- Ante síntomas, lesiones o situaciones médicas, recomendá
  consultar con un profesional habilitado.

ALCANCE

Respondé consultas pedagógicas sobre Reflexología Podal y el
contenido del curso. Si la consulta es completamente ajena,
explicá amablemente cuál es tu función.
"""


# ---------------------------------------------------------
# RUTAS
# ---------------------------------------------------------

@app.get("/", include_in_schema=False)
def home():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "openai_configured": bool(OPENAI_API_KEY),
        "vector_store_configured": bool(
            OPENAI_VECTOR_STORE_ID
        ),
        "model": OPENAI_MODEL,
        "version": "0.5.0",
    }


@app.post("/chat", response_model=ChatResponse)
def chat(data: ChatRequest):
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="OpenAI no está configurado.",
        )

    if not OPENAI_VECTOR_STORE_ID:
        raise HTTPException(
            status_code=503,
            detail="La base de conocimiento no está configurada.",
        )

    question = data.message.strip()

    # Conservamos solo los últimos 10 mensajes para controlar
    # costos y evitar conversaciones excesivamente largas.
    recent_history = data.history[-10:]

    conversation_input = [
        {
            "role": item.role,
            "content": item.content,
        }
        for item in recent_history
    ]

    conversation_input.append(
        {
            "role": "user",
            "content": question,
        }
    )

    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            instructions=SYSTEM_PROMPT,
            input=conversation_input,
            tools=[
                {
                    "type": "file_search",
                    "vector_store_ids": [
                        OPENAI_VECTOR_STORE_ID
                    ],
                    "max_num_results": 5,
                }
            ],
        )

        answer = response.output_text.strip()

        if not answer:
            raise HTTPException(
                status_code=502,
                detail="El modelo no devolvió una respuesta.",
            )

        return ChatResponse(
            answer=answer,
            model=OPENAI_MODEL,
        )

    except AuthenticationError as error:
        logger.exception("Clave inválida: %s", error)

        raise HTTPException(
            status_code=502,
            detail="La clave de OpenAI no es válida.",
        ) from error

    except RateLimitError as error:
        logger.exception("Límite de uso: %s", error)

        raise HTTPException(
            status_code=429,
            detail=(
                "El servicio alcanzó temporalmente "
                "su límite de uso."
            ),
        ) from error

    except APIConnectionError as error:
        logger.exception("Error de conexión: %s", error)

        raise HTTPException(
            status_code=502,
            detail="No fue posible conectarse con OpenAI.",
        ) from error

    except APIStatusError as error:
        logger.exception("Error de OpenAI: %s", error)

        raise HTTPException(
            status_code=502,
            detail="OpenAI devolvió un error.",
        ) from error

    except HTTPException:
        raise

    except Exception as error:
        logger.exception("Error inesperado: %s", error)

        raise HTTPException(
            status_code=500,
            detail=(
                "Ocurrió un error inesperado al generar "
                "la respuesta."
            ),
        ) from error
