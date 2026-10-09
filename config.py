"""Configuración central del proyecto. Nada de rutas ni llaves hardcodeadas fuera de aquí."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

VAULT_PATH = Path(os.environ["VAULT_PATH"])
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "obsidian_vault"

GENERATION_MODEL = "claude-haiku-4-5-20251001"
TOP_K = 5

# HyDE (Hypothetical Document Embeddings): en vez de embeder la pregunta del
# usuario directamente, se genera una respuesta hipotética breve y se embede
# ESA para buscar en el vector store — reduce la brecha de forma lingüística
# entre preguntas (lenguaje natural) y notas (afirmaciones, checkboxes). Ver
# [[Transformación de Preguntas para RAG]].
#
# Medido contra golden_set.json (5/5 pasan con y sin HyDE): no mostró mejora
# medible en este corpus — mismo source_hit en los 5 casos, y en el control
# negativo metió ruido adicional (un chunk irrelevante de más). Sí agrega una
# llamada extra al LLM por query (más costo y latencia). Por eso queda
# desactivado por default: implementado y disponible (pásale use_hyde=True a
# answer_question para probarlo), pero sin evidencia que justifique activarlo
# para este vault — la decisión correcta, dado que no se demostró beneficio,
# es no pagar el costo. Si el vault crece y las preguntas reales se alejan
# más del vocabulario de las notas, vale la pena volver a medir.
USE_HYDE = False

# Regla de seguridad no negociable (ver CLAUDE.md del vault y
# CHIEF/03-PROJECTS/rag-vault-obsidian.md): estas rutas NUNCA se indexan.
EXCLUDED_DIR_NAMES = {
    "Contrasena",
    "Cosas importantes",
    ".obsidian",
    ".git",
    ".claude",
}


def is_excluded(path: Path) -> bool:
    """True si cualquier parte de la ruta cae en una carpeta excluida."""
    return any(part in EXCLUDED_DIR_NAMES for part in path.parts)
