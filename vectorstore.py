"""
Embeddings + Vector Store.

Modelo de embeddings: BAAI/bge-small-en-v1.5 vía FastEmbed (ONNX Runtime, local,
sin torch). Se eligió FastEmbed en vez de sentence-transformers porque esta
máquina es Mac Intel (x86_64) y PyTorch dejó de publicar builds para esa
arquitectura después de la versión 2.2.2 — incompatible con las versiones
recientes de sentence-transformers/transformers. FastEmbed corre sobre ONNX
Runtime, sin esa dependencia, y sigue siendo 100% local y gratis. Si más
adelante la calidad del retrieval no es suficiente, el candidato a cambiar es
Voyage AI (partner oficial de Anthropic para embeddings) u OpenAI
text-embedding-3-small — ver [[Embeddings]] para criterios de selección.

Vector DB: Chroma local (persistida en disco en config.CHROMA_DB_DIR) — elegido
por ser zero-config y Python-native, ideal para prototipos. Ver [[Vector Databases]].

Nota: el vault está en español, así que el modelo de embeddings debe ser
multilingüe (no bge-small-en, que es solo-inglés) — se usa la variante
multilingual de MiniLM.
"""
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma

import config
from ingest import load_and_chunk_vault

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def get_embedding_function() -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)


def build_vectorstore(force_rebuild: bool = False) -> Chroma:
    """
    Si ya existe un índice persistido y no se pide rebuild, lo carga.
    Si no, ingiere + chunkea + embede + persiste desde cero.
    """
    embedding_fn = get_embedding_function()

    db = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embedding_fn,
        persist_directory=config.CHROMA_DB_DIR,
    )

    existing_count = db._collection.count()

    if existing_count > 0 and not force_rebuild:
        print(f"Índice existente encontrado: {existing_count} vectores. Usando el existente.")
        return db

    if existing_count > 0 and force_rebuild:
        print(f"Rebuild forzado: borrando {existing_count} vectores existentes.")
        db.delete_collection()
        db = Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=embedding_fn,
            persist_directory=config.CHROMA_DB_DIR,
        )

    print("Cargando y chunkeando el vault...")
    chunks = load_and_chunk_vault()
    print(f"{len(chunks)} chunks a embeder e indexar (modelo: {EMBEDDING_MODEL})...")

    # Chroma necesita IDs únicos; usamos índice + fuente para que sean estables
    ids = [f"{i}:{doc.metadata.get('source', 'unknown')}" for i, doc in enumerate(chunks)]
    db.add_documents(documents=chunks, ids=ids)

    print(f"Índice construido: {db._collection.count()} vectores persistidos en {config.CHROMA_DB_DIR}/")
    return db


if __name__ == "__main__":
    db = build_vectorstore()

    sample_queries = [
        "¿cuál es mi meta financiera para el próximo año?",
        "¿qué es RAG y cómo funciona?",
        "¿en qué empresa trabajo actualmente?",
    ]

    for query in sample_queries:
        print(f"\n=== Query: {query!r} ===")
        results = db.similarity_search(query, k=3)
        for i, doc in enumerate(results, 1):
            print(f"{i}. [{doc.metadata.get('source')}] {doc.metadata.get('title')}"
                  f" / {doc.metadata.get('H1', '')} > {doc.metadata.get('H2', '')}")
            print(f"   {doc.page_content[:150].strip()}...")
