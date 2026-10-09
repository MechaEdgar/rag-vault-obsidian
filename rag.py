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


MAX_QUERY_LENGTH = 2000


def validate_query(query: str) -> str | None:
    """Input validation básica antes de que la query llegue al retrieval o al
    modelo (ver [[Controles a Nivel de Sistema para AI - Input Validation,
    Filtrado de Instrucciones y MLSecOps]]). No reemplaza ningún guardrail de
    contenido — solo rechaza inputs claramente inválidos antes de gastar una
    llamada a la API. Devuelve un mensaje de error si la query es inválida,
    o None si puede continuar."""
    if not query or not query.strip():
        return "La pregunta no puede estar vacía."
    if len(query) > MAX_QUERY_LENGTH:
        return f"La pregunta es demasiado larga (máximo {MAX_QUERY_LENGTH} caracteres)."
    return None


HYDE_PROMPT = """Genera una respuesta hipotética breve (1-2 oraciones) a la siguiente pregunta, como si fuera el contenido de una nota personal. No te preocupes por la precisión factual — es solo para mejorar una búsqueda semántica, nunca se le muestra al usuario.

Pregunta: {query}"""


def generate_hypothetical_answer(client: Anthropic, query: str) -> str:
    """HyDE: genera una respuesta hipotética para usar en el retrieval en vez
    de la pregunta cruda — reduce la brecha de forma lingüística entre cómo
    se pregunta (lenguaje natural) y cómo están escritas las notas
    (afirmaciones, checkboxes). Ver [[Transformación de Preguntas para RAG]]
    y config.USE_HYDE."""
    response = client.messages.create(
        model=config.GENERATION_MODEL,
        max_tokens=100,
        messages=[{"role": "user", "content": HYDE_PROMPT.format(query=query)}],
    )
    return response.content[0].text.strip()


def format_context(docs) -> str:
    blocks = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "desconocido")
        title = doc.metadata.get("title", "")
        blocks.append(f"[Fuente {i}: {title} ({source})]\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def answer_question(db, client: Anthropic, query: str, k: int = config.TOP_K, use_hyde: bool = config.USE_HYDE):
    validation_error = validate_query(query)
    if validation_error:
        return validation_error, []

    search_query = generate_hypothetical_answer(client, query) if use_hyde else query
    docs = db.similarity_search(search_query, k=k)

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
