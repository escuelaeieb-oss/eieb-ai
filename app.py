import logging
import os

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

if not OPENAI_API_KEY:
    logger.warning(
        "La variable OPENAI_API_KEY no está configurada."
    )

if not OPENAI_VECTOR_STORE_ID:
    logger.warning(
        "La variable OPENAI_VECTOR_STORE_ID no está configurada."
    )

client = (
    OpenAI(api_key=OPENAI_API_KEY)
    if OPENAI_API_KEY
    else None
)


# ---------------------------------------------------------
# APLICACIÓN FASTAPI
# ---------------------------------------------------------

app = FastAPI(
    title="Profesora IA EIEB",
    description=(
        "Asistente pedagógico de la Escuela Iberoamericana "
        "de Estética y Bienestar."
    ),
    version="0.4.0",
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
# MODELOS DE DATOS
# ---------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Pregunta pedagógica realizada por la alumna.",
        examples=["¿Qué es la reflexología podal?"],
    )

    previous_response_id: str | None = Field(
        default=None,
        description=(
            "Identificador de la respuesta anterior para "
            "mantener el contexto de la conversación."
        ),
    )


class ChatResponse(BaseModel):
    answer: str
    model: str
    response_id: str


# ---------------------------------------------------------
# INSTRUCCIONES DE LA PROFESORA
# ---------------------------------------------------------

SYSTEM_PROMPT = """
Sos la Profesora IA oficial de la Escuela Iberoamericana de
Estética y Bienestar, EIEB.

En esta primera versión estás especializada exclusivamente en
Reflexología Podal y acompañás a alumnas que ya compraron el curso.

PRIORIDAD DE INFORMACIÓN

- Consultá primero los documentos oficiales del curso mediante
  la herramienta File Search.
- Cuando la respuesta esté en los documentos, basate principalmente
  en ellos y no agregues datos externos innecesarios.
- Si los documentos no contienen información suficiente, decilo
  claramente.
- Podés complementar con conocimiento general confiable, pero
  debés aclarar que se trata de información complementaria.
- Nunca afirmes que algo aparece en una página, módulo o clase
  específica si no encontraste esa referencia.
- Tené en cuenta los mensajes anteriores de la conversación.
- Cuando la alumna diga "esa parte", "la tercera", "lo anterior"
  o expresiones similares, interpretalas usando el contexto previo.

OBJETIVO PEDAGÓGICO

Tu objetivo no es solamente responder preguntas. Tu objetivo es
que la alumna comprenda el contenido, gane confianza y pueda
continuar estudiando.

FORMA DE RESPONDER

- Respondé siempre en español.
- Usá un tono amable, profesional, paciente y cercano.
- Explicá con palabras sencillas.
- Cuando uses un término técnico, explicá qué significa.
- Dividí las explicaciones complejas en pasos.
- Utilizá ejemplos cuando ayuden a comprender.
- Evitá respuestas innecesariamente extensas.
- No respondas de forma fría o robótica.
- No llenes la respuesta de emojis.
- No termines siempre con una pregunta.
- Cuando sea apropiado, ofrecé explicar el tema de una forma más
  sencilla o con un ejemplo práctico.
- Si la alumna dice que no entendió, no repitas exactamente la
  misma respuesta: explicá el tema de otra manera.

SEGURIDAD

- No diagnostiques enfermedades.
- No indiques tratamientos médicos.
- No afirmes que la reflexología cura enfermedades.
- No prometas resultados terapéuticos.
- No reemplaces la consulta con un profesional de la salud.
- Ante síntomas, lesiones o situaciones médicas, recomendá
  consultar con un profesional habilitado.

ALCANCE

- Respondé consultas pedagógicas sobre Reflexología Podal y el
  contenido del curso.
- Si preguntan algo completamente ajeno, explicá amablemente que
  tu función es acompañar el aprendizaje dentro de EIEB.
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
        "version": "0.4.0",
    }


@app.post("/chat", response_model=ChatResponse)
def chat(data: ChatRequest):
    if client is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "La conexión con OpenAI no está configurada."
            ),
        )

    if not OPENAI_VECTOR_STORE_ID:
        raise HTTPException(
            status_code=503,
            detail=(
                "La base de conocimiento no está configurada."
            ),
        )

    question = data.message.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="La pregunta no puede estar vacía.",
        )

    try:
        request_data = {
            "model": OPENAI_MODEL,
            "instructions": SYSTEM_PROMPT,
            "input": question,
            "tools": [
                {
                    "type": "file_search",
                    "vector_store_ids": [
                        OPENAI_VECTOR_STORE_ID
                    ],
                    "max_num_results": 5,
                }
            ],
        }

        if data.previous_response_id:
            request_data["previous_response_id"] = (
                data.previous_response_id
            )

        response = client.responses.create(**request_data)

        answer = response.output_text.strip()

        if not answer:
            raise HTTPException(
                status_code=502,
                detail=(
                    "El modelo no devolvió una respuesta."
                ),
            )

        return ChatResponse(
            answer=answer,
            model=OPENAI_MODEL,
            response_id=response.id,
        )

    except AuthenticationError as error:
        logger.exception(
            "Clave de OpenAI inválida: %s",
            error,
        )

        raise HTTPException(
            status_code=502,
            detail="La clave de OpenAI no es válida.",
        ) from error

    except RateLimitError as error:
        logger.exception(
            "Límite o saldo insuficiente en OpenAI: %s",
            error,
        )

        raise HTTPException(
            status_code=429,
            detail=(
                "El servicio alcanzó temporalmente "
                "su límite de uso."
            ),
        ) from error

    except APIConnectionError as error:
        logger.exception(
            "No se pudo conectar con OpenAI: %s",
            error,
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "No fue posible conectarse con OpenAI."
            ),
        ) from error

    except APIStatusError as error:
        logger.exception(
            "OpenAI devolvió un error de estado: %s",
            error,
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "OpenAI devolvió un error al procesar "
                "la conversación."
            ),
        ) from error

    except HTTPException:
        raise

    except Exception as error:
        logger.exception(
            "Error inesperado en /chat: %s",
            error,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Ocurrió un error inesperado al generar "
                "la respuesta."
            ),
        ) from error
