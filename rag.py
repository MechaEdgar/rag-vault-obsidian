"""
Retrieval + Generación.

Pipeline: query del usuario → top-k retrieval en Chroma → se pasa el contexto
recuperado a Claude → responde en español citando las notas fuente.

El prompt instruye explícitamente a NO alucinar: si el contexto no alcanza
para responder, debe decirlo en vez de inventar. Esto es RAG "naive" (sin
re-ranking ni filtro de relevancia por score todavía) — ver [[Arquitectura RAG]]
para la distinción naive vs advanced. Mejora pendiente si el retrieval trae
ruido: filtrar por score de similitud antes de pasarle el contexto al LLM.
"""
from anthropic import Anthropic

import config
from vectorstore import build_vectorstore

SYSTEM_PROMPT = """Eres un asistente que responde preguntas sobre las notas personales de un vault de Obsidian, usando ÚNICAMENTE el contexto que se te proporciona en cada mensaje.

Reglas:
- Responde en español, de forma clara y directa.
- Si el contexto tiene información suficiente, respóndela y al final agrega una línea "Fuentes:" listando los títulos de las notas que usaste.
- Si el contexto NO tiene información suficiente para responder la pregunta, dilo explícitamente: "No encontré información suficiente en el vault para responder esto." No inventes datos ni fuentes que no estén en el contexto."""


def format_context(docs) -> str:
    blocks = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "desconocido")
        title = doc.metadata.get("title", "")
        blocks.append(f"[Fuente {i}: {title} ({source})]\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def answer_question(db, client: Anthropic, query: str, k: int = config.TOP_K):
    docs = db.similarity_search(query, k=k)

    if not docs:
        return "No encontré información relevante en el vault para responder esto.", []

    context = format_context(docs)
    user_message = f"Contexto recuperado del vault:\n\n{context}\n\n---\n\nPregunta: {query}"

    response = client.messages.create(
        model=config.GENERATION_MODEL,
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    answer = response.content[0].text
    sources = sorted({doc.metadata.get("source", "desconocido") for doc in docs})
    return answer, sources


if __name__ == "__main__":
    print("Cargando índice...")
    db = build_vectorstore()
    client = Anthropic(api_key=config.ANTHROPIC_API_KEY)

    print("\nRAG sobre el vault listo. Escribe una pregunta ('salir' para terminar).\n")
    while True:
        query = input("> ").strip()
        if query.lower() in {"salir", "exit", "quit"}:
            break
        if not query:
            continue

        answer, sources = answer_question(db, client, query)
        print(f"\n{answer}\n")
        if sources:
            print("(chunks recuperados de:", ", ".join(sources), ")")
        print()
