"""
Ingesta y chunking de las notas del vault de Obsidian.

Pipeline:
  1. Recorrer el vault y encontrar todos los .md, EXCLUYENDO rutas sensibles
     (ver config.EXCLUDED_DIR_NAMES — Contrasena, Cosas importantes, etc.)
  2. Separar frontmatter (YAML) del cuerpo de cada nota
  3. Chunkear: primero por headers de Markdown (document-aware), y si una
     sección queda muy grande, subdividirla con recursive splitting.
     Ver CHIEF/02-CONNECTIONS/AI Engineering/RAG/Estrategias de Chunking.md
  4. Adjuntar metadata a cada chunk: ruta relativa, título, tags, headers.

Este módulo expone load_and_chunk_vault() para que el paso de embeddings
(siguiente etapa) lo importe directamente, sin pasar por un archivo intermedio.
"""
import yaml
from pathlib import Path

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document

import config

HEADERS_TO_SPLIT_ON = [("#", "H1"), ("##", "H2"), ("###", "H3")]

# ~500 tokens de la regla práctica de Estrategias de Chunking ≈ 2000 caracteres
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200  # 10% de CHUNK_SIZE


def find_markdown_files(vault_path: Path) -> list[Path]:
    """Todos los .md del vault, excluyendo las rutas sensibles de config.py."""
    all_md = list(vault_path.rglob("*.md"))
    return [p for p in all_md if not config.is_excluded(p.relative_to(vault_path))]


def split_frontmatter(text: str) -> tuple[dict, str]:
    """Separa el frontmatter YAML (si existe) del cuerpo de la nota."""
    if not text.startswith("---"):
        return {}, text

    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text

    _, raw_frontmatter, body = parts
    try:
        metadata = yaml.safe_load(raw_frontmatter) or {}
    except yaml.YAMLError:
        metadata = {}
    return metadata, body.strip()


def chunk_note(path: Path, vault_path: Path) -> list[Document]:
    """Convierte una nota en 1+ Document con metadata, listos para embeddings."""
    raw_text = path.read_text(encoding="utf-8")
    frontmatter, body = split_frontmatter(raw_text)

    relative_path = str(path.relative_to(vault_path))
    title = frontmatter.get("title") or path.stem
    tags = frontmatter.get("tags", [])

    base_metadata = {
        "source": relative_path,
        "title": title,
        "tags": ", ".join(tags) if isinstance(tags, list) else str(tags),
    }

    if not body.strip():
        return []

    # Paso 1: split por headers de Markdown (document-aware)
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON)
    header_chunks = header_splitter.split_text(body)

    if not header_chunks:
        header_chunks = [Document(page_content=body, metadata={})]

    # Paso 2: si una sección es muy grande, subdividir recursivamente
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    final_chunks = []
    for chunk in header_chunks:
        sub_chunks = recursive_splitter.split_text(chunk.page_content)
        for sub in sub_chunks:
            if not sub.strip():
                continue
            metadata = {**base_metadata, **chunk.metadata}

            # Contextual retrieval (simplificado): el título y los headers que
            # MarkdownHeaderTextSplitter separó a metadata no quedan en el texto
            # que se embede. Sin esto, una nota como "Tareas Diarias" (kanban de
            # hábitos) puede perder frente a notas que solo HABLAN de "tareas
            # diarias" como concepto, aunque el contenido real sea otro.
            header_path = " > ".join(
                v for k, v in chunk.metadata.items() if k.startswith("H") and v
            )
            context_prefix = title if not header_path else f"{title} — {header_path}"
            contextualized = f"{context_prefix}\n\n{sub}"

            final_chunks.append(Document(page_content=contextualized, metadata=metadata))

    return final_chunks


def load_and_chunk_vault(vault_path: Path | None = None) -> list[Document]:
    vault_path = vault_path or config.VAULT_PATH
    md_files = find_markdown_files(vault_path)

    all_chunks = []
    for path in md_files:
        all_chunks.extend(chunk_note(path, vault_path))

    return all_chunks


if __name__ == "__main__":
    vault_path = config.VAULT_PATH
    total_md_files = len(list(vault_path.rglob("*.md")))
    ingested_files = find_markdown_files(vault_path)
    excluded_count = total_md_files - len(ingested_files)

    print(f"Total .md en el vault: {total_md_files}")
    print(f"Excluidos (Contrasena/Cosas importantes/.obsidian/.git/.claude): {excluded_count}")
    print(f"Archivos a ingerir: {len(ingested_files)}")

    # Chequeo de seguridad explícito antes de chunkear nada
    for p in ingested_files:
        rel = p.relative_to(vault_path)
        assert not config.is_excluded(rel), f"¡Fuga de seguridad! no debería ingerirse: {rel}"
    print("✅ Verificado: ningún archivo de carpetas excluidas entró a la lista.")

    chunks = load_and_chunk_vault(vault_path)
    print(f"\nTotal de chunks generados: {len(chunks)}")
    if chunks:
        print("\n--- Ejemplo de chunk ---")
        print("Metadata:", chunks[0].metadata)
        print("Contenido (primeros 200 chars):", chunks[0].page_content[:200])
