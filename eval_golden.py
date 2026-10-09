"""
Evaluación automática contra un golden dataset — el equivalente de un CI check
para este proyecto (ver [[CI para Agentes LLM - Golden Datasets y Pinning de
Versiones]] en el vault). Antes, validar el RAG significaba leer 5 respuestas
a mano y juzgar "se ve bien" — esto las corre automáticamente y da un score
reproducible, para poder volver a correrlo después de cualquier cambio
(modelo de embedding nuevo, chunking nuevo, prompt nuevo) sin tener que
re-leer todo a ojo.

Separación pública/privada: golden_set.json (gitignored) tiene las preguntas
reales con datos personales del vault. golden_set.example.json (committeado)
muestra el schema con ejemplos genéricos — mismo patrón que .env/.env.example
ya usado en este proyecto.

Chequeos por pregunta:
- ¿Al menos una de las fuentes esperadas aparece en las fuentes citadas?
- Keyword overlap entre la respuesta y las keywords esperadas (métrica básica,
  no mide calidad semántica — ver advertencia en el resumen final).
- Para controles negativos: ¿la respuesta contiene la frase de rechazo
  esperada, en vez de alucinar una respuesta?
"""
import json
import sys
from pathlib import Path

import config
from rag import answer_question
from vectorstore import build_vectorstore
from anthropic import Anthropic

REFUSAL_PHRASE = "no encontré información"

GOLDEN_SET_PATH = Path("golden_set.json")
GOLDEN_SET_EXAMPLE_PATH = Path("golden_set.example.json")


def load_golden_set() -> list[dict]:
    if GOLDEN_SET_PATH.exists():
        path = GOLDEN_SET_PATH
    else:
        print(
            f"⚠️  No existe {GOLDEN_SET_PATH} — usando {GOLDEN_SET_EXAMPLE_PATH} "
            "(solo 2 ejemplos genéricos, no tu golden set real).\n"
            "   Copia golden_set.example.json a golden_set.json y rellénalo con "
            "tus propias preguntas/respuestas esperadas.\n"
        )
        path = GOLDEN_SET_EXAMPLE_PATH
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def keyword_overlap(answer: str, expected_keywords: list[str]) -> float:
    """Métrica básica: fracción de keywords esperadas presentes en la respuesta
    (case-insensitive). No mide calidad semántica — un score bajo no prueba que
    la respuesta esté mal, solo que usó fraseo distinto. Ver [[Evaluaciones de
    AI Safety - Benchmarks y la Brecha Humano-Judge]] para esta misma advertencia
    aplicada a la evaluación de Claude con keyword overlap en el libro."""
    if not expected_keywords:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return hits / len(expected_keywords)


def run_case(db, client, case: dict) -> dict:
    answer, sources = answer_question(db, client, case["query"])
    result = {"id": case["id"], "query": case["query"], "answer": answer, "sources": sources}

    if case.get("negative_control"):
        result["pass"] = REFUSAL_PHRASE in answer.lower()
        result["check"] = "rechazo esperado (control negativo)"
        return result

    expected_sources = case.get("expected_sources_contains", [])
    source_hit = (
        not expected_sources
        or any(exp.lower() in src.lower() for exp in expected_sources for src in sources)
    )
    kw_score = keyword_overlap(answer, case.get("expected_keywords", []))

    result["source_hit"] = source_hit
    result["keyword_score"] = round(kw_score, 2)
    result["pass"] = source_hit and kw_score >= 0.5
    result["check"] = "fuente esperada + keyword overlap >= 0.5"
    return result


def main():
    cases = load_golden_set()
    print(f"Cargando índice y corriendo {len(cases)} casos del golden set...\n")

    db = build_vectorstore()
    client = Anthropic(api_key=config.ANTHROPIC_API_KEY)

    results = [run_case(db, client, case) for case in cases]

    for r in results:
        status = "✅ PASS" if r["pass"] else "❌ FAIL"
        print(f"{status}  [{r['id']}] {r['query']}")
        print(f"        chequeo: {r['check']}")
        if "source_hit" in r:
            print(f"        fuentes citadas: {r['sources']} (esperada presente: {r['source_hit']})")
            print(f"        keyword overlap: {r['keyword_score']}")
        print()

    passed = sum(1 for r in results if r["pass"])
    print(f"Resultado: {passed}/{len(results)} casos pasaron.")
    if passed < len(results):
        print(
            "\n⚠️  Un score de keyword overlap bajo no prueba necesariamente una "
            "respuesta mala — puede ser fraseo distinto. Revisa la respuesta "
            "completa de los casos que fallaron antes de asumir una regresión real."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
