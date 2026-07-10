import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

if not OPENAI_API_KEY:
    logger.warning(
        "La variable OPENAI_API_KEY no está configurada."
    )

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ---------------------------------------------------------
# APLICACIÓN FASTAPI
# ---------------------------------------------------------

app = FastAPI(
    title="Profesora IA EIEB",
    description=(
        "Backend del asistente pedagógico de la Escuela "
        "Iberoamericana de Estética y Bienestar."
    ),
    version="0.2.0",
)


# Para el MVP permitimos solicitudes desde cualquier origen.
# Más adelante lo limitaremos al dominio exacto de Tiendup.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
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


class ChatResponse(BaseModel):
    answer: str
    model: str


# ---------------------------------------------------------
# INSTRUCCIONES DE LA PROFESORA
# ---------------------------------------------------------

SYSTEM_PROMPT = """
Sos la Profesora IA oficial de la Escuela Iberoamericana de
Estética y Bienestar, EIEB.

En esta primera versión estás especializada en Reflexología Podal
y acompañás únicamente a alumnas que ya compraron el curso.

OBJETIVO PRINCIPAL

Tu objetivo no es solamente responder preguntas. Tu objetivo es
que la alumna comprenda el contenido, gane confianza y pueda
continuar estudiando.

FORMA DE RESPONDER

- Respondé siempre en español.
- Utilizá un tono amable, profesional, paciente y cercano.
- Explicá con palabras sencillas.
- Cuando uses un término técnico, explicá qué significa.
- Dividí las explicaciones complejas en pasos.
- Utilizá ejemplos cuando ayuden a comprender.
- Evitá respuestas innecesariamente extensas.
- No respondas de forma fría o robótica.
- No llenes la respuesta de emojis.
- No termines siempre con una pregunta; hacelo únicamente cuando
  ayude a continuar el aprendizaje.

PRECISIÓN

- No inventes información.
- Si no contás con información suficiente, decilo claramente.
- Diferenciá entre información educativa y asesoramiento
  profesional personalizado.
- Cuando existan varios enfoques, explicá que pueden existir
  distintas escuelas o metodologías.

SEGURIDAD

- No diagnostiques enfermedades.
- No indiques tratamientos médicos.
- No afirmes que la reflexología cura enfermedades.
- No prometas resultados terapéuticos.
- No reemplaces la consulta con un profesional de la salud.
- Ante síntomas, lesiones o situaciones médicas, recomendá
  consultar con un profesional habilitado.

ALCANCE

- Priorizá consultas pedagógicas relacionadas con Reflexología,
  estética, bienestar y el contenido del curso.
- Si preguntan algo completamente ajeno, explicá amablemente que
  tu función es acompañar el aprendizaje dentro de EIEB.
- Todavía no afirmes que una respuesta proviene de un PDF o módulo
  específico, porque la base documental se conectará en la
  siguiente etapa.
"""


# ---------------------------------------------------------
# RUTAS
# ---------------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Profesora IA de EIEB funcionando",
        "version": "0.2.0",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "openai_configured": bool(OPENAI_API_KEY),
        "model": OPENAI_MODEL,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(data: ChatRequest):
    if client is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "La conexión con OpenAI todavía no está "
                "configurada."
            ),
        )

    question = data.message.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="La pregunta no puede estar vacía.",
        )

    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            instructions=SYSTEM_PROMPT,
            input=question,
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
        logger.exception("Clave de OpenAI inválida: %s", error)

        raise HTTPException(
            status_code=502,
            detail=(
                "La clave de OpenAI no es válida o no tiene "
                "permisos."
            ),
        ) from error

    except RateLimitError as error:
        logger.exception(
            "Límite o saldo insuficiente en OpenAI: %s",
            error,
        )

        raise HTTPException(
            status_code=429,
            detail=(
                "El servicio alcanzó temporalmente su límite "
                "de uso. Intentá nuevamente más tarde."
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
                "No fue posible conectarse con el servicio "
                "de inteligencia artificial."
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
                "El servicio de inteligencia artificial "
                "devolvió un error."
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
