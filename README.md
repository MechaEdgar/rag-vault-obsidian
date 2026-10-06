# RAG sobre un Vault de Obsidian

Un sistema RAG (Retrieval-Augmented Generation) completo que responde preguntas sobre un vault de notas personales en Markdown (Obsidian), citando siempre la nota fuente y negándose a contestar cuando no tiene información suficiente.

Construido como proyecto de portfolio para AI Orchestration Engineering — el objetivo no era solo "conectar un vector store con un LLM", sino pasar por el ciclo completo de diseño, decisiones de infraestructura, y debugging de un problema real de retrieval.

## Qué hace

```
$ python rag.py
Cargando índice...

RAG sobre el vault listo. Escribe una pregunta ('salir' para terminar).

> ¿cuál es la diferencia entre fixed-size y recursive splitting?
Fixed-size divide el texto cada N caracteres fijos, sin importar la
estructura — puede cortar oraciones a la mitad. Recursive splitting
intenta respetar separadores naturales (párrafos, saltos de línea,
puntos) antes de recurrir al corte forzado...

Fuentes: Estrategias de Chunking
```

Si la pregunta no tiene respuesta en el vault, lo dice explícitamente en vez de inventar una respuesta o una fuente falsa.

## Arquitectura

```
Notas .md del vault
        │
        ▼
┌───────────────────┐
│  1. Ingesta        │  Recorre el vault, EXCLUYE carpetas sensibles,
│     (ingest.py)    │  separa frontmatter YAML del cuerpo
└─────────┬──────────┘
          ▼
┌───────────────────┐
│  2. Chunking        │  Document-aware (split por headers H1-H3) +
│     (ingest.py)     │  recursive splitting si una sección es grande.
│                      │  Cada chunk se prefija con título+headers antes
│                      │  de embederse (contextual retrieval)
└─────────┬──────────┘
          ▼
┌───────────────────┐
│  3. Embeddings      │  FastEmbed (ONNX, local, sin costo),
│     (vectorstore.py)│  modelo multilingüe
└─────────┬──────────┘
          ▼
┌───────────────────┐
│  4. Vector Store    │  Chroma, persistido en disco
│     (vectorstore.py)│
└─────────┬──────────┘
          ▼
┌───────────────────┐
│  5. Retrieval +     │  Top-k similarity search → contexto → Claude
│     Generación       │  genera la respuesta citando fuentes, o dice
│     (rag.py)         │  "no encontré información" si no aplica
└───────────────────┘
```

## Decisiones de diseño (y por qué)

**Chunking: document-aware + recursive, con contexto inyectado.** Las notas de Obsidian ya tienen estructura semántica (headers, secciones) — dividir por ahí preserva mejor el significado que un corte por tamaño fijo. Cuando una sección es muy grande, se subdivide recursivamente (500 tokens / 10% overlap como punto de partida). Cada chunk se prefija con el título de la nota y su ruta de headers antes de embederse — sin esto, una nota que *menciona* un tema de pasada puede ganarle en similitud semántica a la nota que *es* ese tema, simplemente porque usa vocabulario más parecido a la pregunta. Ver la sección de debugging abajo — así se descubrió.

**Embeddings: FastEmbed en vez de sentence-transformers/PyTorch.** La intención original era usar un modelo de Hugging Face vía `sentence-transformers`. En una Mac con arquitectura x86_64 (o corriendo bajo Rosetta), PyTorch dejó de publicar builds después de la 2.2.2, incompatible con versiones recientes de `transformers`. En vez de fijar versiones viejas y frágiles, se usó **FastEmbed** (ONNX Runtime, mantenido por Qdrant) — mismo resultado (embeddings locales, gratis, sin llamadas a APIs externas), sin la dependencia de PyTorch. Modelo: `paraphrase-multilingual-MiniLM-L12-v2`, elegido porque el corpus está en español.

**Vector store: Chroma local.** Zero-config, embebido, ideal para un proyecto de este tamaño (cientos de chunks, no millones). Para producción a escala, los candidatos naturales serían Pinecone (managed) o pgvector (si ya hay Postgres).

**Generación: Claude Haiku.** Modelo pequeño y económico — suficiente para resumir contexto recuperado y citar fuentes, que no requiere razonamiento profundo. El prompt del sistema prohíbe explícitamente responder fuera del contexto proporcionado.

**Seguridad: exclusión de rutas por diseño, no por honor.** El vault de origen tiene carpetas con credenciales y documentos sensibles. El filtro de exclusión vive en `config.py`, se aplica antes de que cualquier archivo llegue al chunking, y se verifica con un assert explícito en cada corrida de `ingest.py` — no es una promesa informal, es una verificación automática que falla la ejecución si algo se escapa.

## Un bug real de retrieval (y cómo se encontró)

Durante la validación, la pregunta *"¿qué hábitos o tareas diarias tengo pendientes?"* devolvía notas que **hablaban sobre** el concepto de "tareas diarias" (documentación de un sistema de workflows), en vez de la nota que **era** la lista real de hábitos.

**Diagnóstico:** el chunking separaba el título y los headers de cada nota a metadata — ese texto nunca llegaba al embedding, solo el cuerpo plano. La nota con la lista real de hábitos no contenía la frase "tareas diarias" en su texto (son solo checkboxes de acciones concretas), mientras que la documentación del sistema sí usaba esa frase repetidamente. El embedding terminaba pareciéndose más a la pregunta por pura coincidencia de vocabulario, no de contenido real.

**Fix:** prefijar cada chunk con `título — ruta de headers` antes de embederlo (una versión simplificada de lo que Anthropic documenta como *Contextual Retrieval*). Con el fix, la nota correcta pasó a ser el resultado top. Se reconstruyó el índice completo y se re-verificaron todas las preguntas de validación anteriores para confirmar que no hubo regresión.

Este es el tipo de problema que no aparece hasta que se prueba con preguntas reales — justo por eso el proyecto incluye una fase de validación con preguntas de dominios distintos, no solo una demo feliz de un solo tipo de pregunta.

## Validación

5 preguntas de dominios distintos sobre el vault real, incluyendo un control negativo (pregunta sin respuesta posible, para confirmar que el sistema no alucina). Las 5 pasaron — una de ellas falló en el primer intento (el bug de arriba), se diagnosticó y corrigió.

## Cómo correrlo

```bash
git clone <este-repo>
cd rag-vault-obsidian
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# editar .env: ANTHROPIC_API_KEY real + VAULT_PATH a tu vault de Obsidian

python ingest.py        # valida la ingesta y el filtro de exclusión
python vectorstore.py   # construye el índice y corre unas queries de prueba
python rag.py           # chat interactivo
```

**Requisito:** una API key de Anthropic (console.anthropic.com) con saldo — el embedding es gratis/local, solo la generación consume la API.

## Limitaciones conocidas / próximos pasos

- Sin re-ranking ni búsqueda híbrida (keyword + vectorial) — RAG "naive". El fix de contextual retrieval ayuda, pero un re-ranker mejoraría más la precisión en corpus más grandes.
- Sin filtro de relevancia por score antes de pasar contexto al LLM — depende del prompt para decir "no sé", no hay un umbral duro.
- Sin evaluación automatizada (el pipeline de 5 preguntas fue manual) — el siguiente paso natural es un pequeño eval set versionado.
- Ratio de ~7.6 chunks por nota es alto para notas cortas; fusionar secciones muy pequeñas podría mejorar la señal.

## Stack

Python · LangChain · Chroma · FastEmbed (ONNX) · Anthropic Claude
