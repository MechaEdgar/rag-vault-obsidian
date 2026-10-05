"""Configuración central del proyecto. Nada de rutas ni llaves hardcodeadas fuera de aquí."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

VAULT_PATH = Path(os.environ["VAULT_PATH"])
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "obsidian_vault"

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
