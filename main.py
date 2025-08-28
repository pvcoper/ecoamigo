import os
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from collections import deque, defaultdict
import re as _re

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from openai import OpenAI

# =============================
# Carga de entorno
# =============================
load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SHOP_NAME      = os.getenv("SHOP_NAME", "GrinHaus")
SHOP_TONE      = os.getenv("SHOP_TONE", "amable, cercano, experto en vida sustentable y productos ecológicos del hogar")
CATALOG_PATH   = os.getenv("CATALOG_PATH", "products.json")
LOG_DIR        = os.getenv("LOG_DIR", "logs")

# Para correr local:
PORT = int(os.getenv("PORT", "8001"))

# =============================
# Cargar catálogo
# =============================
def load_catalog(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return []
    except Exception:
        return []

CATALOG = load_catalog(CATALOG_PATH)

# =============================
# App FastAPI + CORS
# =============================
app = FastAPI(title="EcoAmigo API", version="2.0")

allowed = os.getenv("ALLOWED_ORIGINS", "")
origins = [o.strip() for o in allowed.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],   # en prod, idealmente NO "*"
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# =============================
# Modelos de request/response
# =============================
class ChatPayload(BaseModel):
    message: Optional[str] = None
    prompt: Optional[str]  = None
    query: Optional[str]   = None
    input_text: Optional[str] = None
    session_id: Optional[str] = None

# =============================
# Memoria en RAM por sesión
# =============================
MAX_TURNS = 6  # últimas 6 interacciones (user+assistant) = 12 mensajes
MEMORY: Dict[str, deque] = defaultdict(lambda: deque(maxlen=2 * MAX_TURNS))

# =============================
# Utilidades
# =============================
def ensure_logs():
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    return Path(LOG_DIR) / "chat_logs.csv"

def log_chat(session_id: str, user_msg: str, answer: str):
    try:
        csv_path = ensure_logs()
        is_new = not csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if is_new:
                w.writerow(["timestamp", "session_id", "user_message", "answer"])
            w.writerow([datetime.now().isoformat(), session_id, user_msg, answer])
    except Exception:
        pass

def match_products(user_text: str, topk: int = 3) -> List[Dict[str, Any]]:
    """
    Búsqueda simple por palabras clave sobre title/tags + filtro de exclusiones por producto.
    Si 'exclude_keywords' del producto aparece en la consulta, el producto se descarta.
    - Palabra suelta: usa borde de palabra (\b ... \b)
    - Frase (tiene espacio): usa coincidencia por subcadena
    """
    if not user_text or not CATALOG:
        return []
    q = user_text.lower()
    qs = set(q.split())

    def is_excluded(prod: Dict[str, Any]) -> bool:
        ex = prod.get("exclude_keywords") or []
        for kw in ex:
            if not kw:
                continue
            kw_norm = str(kw).lower().strip()
            if not kw_norm:
                continue
            # Si es una frase (tiene espacio), usamos subcadena simple
            if " " in kw_norm:
                if kw_norm in q:
                    return True
            else:
                # Palabra suelta → borde de palabra
                if _re.search(rf"\b{_re.escape(kw_norm)}\b", q):
                    return True
        return False

    scored = []
    for p in CATALOG:
        # 1) Filtro de exclusión por producto
        if is_excluded(p):
            continue

        # 2) Scoring básico por coincidencias en título/tags
        title = (p.get("title") or "").lower()
        tags  = " ".join(p.get("tags", [])).lower() if isinstance(p.get("tags"), list) else str(p.get("tags") or "").lower()
        hay   = f"{title} {tags}"
        score = sum(1 for token in qs if token and token in hay)

        if score > 0:
            scored.append((score, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:topk]]

def render_product_cards(products: List[Dict[str, Any]]) -> str:
    """Bloque HTML con tarjetas y botones."""
    if not products:
        return ""
    cards_html = []
    for p in products:
        title = p.get("title", "Producto")
        url   = p.get("url", "#")
        price = p.get("price", "")
        vid   = p.get("variant_id", "")
        cards_html.append(f"""
<div class="eco-card" style="border:1px solid #e5e7eb;border-radius:10px;padding:10px;margin:8px 0;background:#ffffff">
  <div style="font-weight:600;margin-bottom:6px;">{title}</div>
  <div style="color:#334155;margin-bottom:6px;">Precio: {price}</div>
  <div style="display:flex;gap:8px;flex-wrap:wrap">
    <button data-variant-id="{vid}" style="padding:8px 12px;border:none;border-radius:8px;background:#16a34a;color:#fff;cursor:pointer">Agregar al carrito</button>
    <a href="{url}" target="_blank" style="padding:8px 12px;border:1px solid #cbd5e1;border-radius:8px;color:#0b4a6e;text-decoration:none;display:inline-block">Ver producto</a>
  </div>
</div>
        """.strip())
    html_block = "<div class=\"eco-cards\">" + "\n".join(cards_html) + "</div>"
    return html_block

def build_client() -> Optional[OpenAI]:
    if not OPENAI_API_KEY:
        return None
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        return client
    except Exception:
        return None

# =============================
# Detección de intención de recomendar (matizada para CL)
# =============================
YES_WORDS = {
    "si","sí","dale","ok","okey","de acuerdo","perfecto","claro","bueno","vale","ya","hagámoslo","hágamoslo"
}

# Saludos/pequeños socializadores que NO deben gatillar recomendación
GREETING_PATTERNS = _re.compile(
    r"\b(hola|buenas|buenos\s+d[ií]as|buenas\s+tardes|buenas\s+noches|"
    r"qué\s+tal|como\s+est[aá]s|soy\s+\w+)\b",
    flags=_re.IGNORECASE
)

# Preguntas / pedidos explícitos de recomendación (imperativos y frases comunes, con variantes locales)
EXPLICIT_RECO_PATTERNS = _re.compile(
    r"(recom(i|e)nd(a|ame|ar|aciones?)|"
    r"¿\s*qu[eé]\s+me\s+recomiendas|"
    r"puedes\s+recomendar(me)?|"
    r"me\s+recom(i|e)endas|"
    r"sug(e|i)r(e|e)?(me)?|suger(e|i)ncia(s)?|"
    r"puedes\s+sugerir(me)?|me\s+sugieres|"
    r"opciones?|alternativa(s)?|"
    r"quiero\s+(ver\s+)?(opciones|productos?|comprar)|"
    r"busco|estoy\s+buscando|ando\s+necesitando|necesito|"
    r"qué\s+opciones|qué\s+alternativas|qué\s+productos|qué\s+hay\s+para|"
    r"tienes\s+algo|qué\s+tienen|muestran\s+productos|"
    r"dame\s+un\s+dato|alg[uú]n\s+dato|datito(s)?|"
    r"dame\s+un\s+tip|alg[uú]n\s+tip|"
    r"dame\s+una\s+idea|consejito(s)?)",
    flags=_re.IGNORECASE
)

# Palabras de compra/acción que también cuentan como intención
PURCHASE_INTENT_PATTERNS = _re.compile(
    r"\b(comprar|compra|precio|cu[eé]sta|vale|agregar|a[nñ]adir|carrito|ver\s+producto|cotizar)\b",
    flags=_re.IGNORECASE
)

def normalize(t: str) -> str:
    return (t or "").strip()

def said_yes(t: str) -> bool:
    t = " " + normalize(t).lower() + " "
    return any((" " + w + " ") in t for w in YES_WORDS)

def is_greeting_only(t: str) -> bool:
    t = normalize(t)
    # Consideramos “solo saludo/presentación” si es corto y calza con saludos
    return (len(t) <= 25) and bool(GREETING_PATTERNS.search(t))

def has_intent_now(t: str) -> bool:
    t = normalize(t)
    if is_greeting_only(t):
        return False
    if EXPLICIT_RECO_PATTERNS.search(t):
        return True
    if PURCHASE_INTENT_PATTERNS.search(t):
        return True
    return False

def assistant_offered_recommendation(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    return (
        "¿quieres que te recomiende" in t
        or "¿te sugiero" in t
        or "¿quieres que te muestre" in t
        or "¿quieres recomendaciones" in t
    )

def should_recommend(user_msg: str, history: deque) -> bool:
    # 1) Intención explícita en el mensaje actual
    if has_intent_now(user_msg):
        return True

    # 2) Confirmación tras oferta previa del asistente
    if history:
        last_assistant = None
        for m in reversed(history):
            if m["role"] == "assistant":
                last_assistant = m["content"]
                break
        if last_assistant and assistant_offered_recommendation(last_assistant):
            return said_yes(user_msg)

    return False

# =============================
# Prompt del sistema
# =============================
SYSTEM_PROMPT = f"""
Eres EcoAmigo, asistente de {SHOP_NAME}. Tu tono es {SHOP_TONE}.
- Mantén el contexto de la conversación.
- Eres experto en sustentabilidad para el hogar.
- Solo responde preguntas relacionadas con ese contexto; si te preguntan algo fuera, redirige suavemente.
- Responde claro y útil.
- MUY IMPORTANTE: Solo recomienda productos si el usuario lo pide explícitamente o si el contexto lo amerita (p. ej., está buscando una solución concreta).
  Si el usuario solo saluda o se presenta, responde cordial y pregunta qué necesita antes de sugerir productos.
- Cuando recomiendes, NO incluyas precios, enlaces ni HTML en el texto libre.
  El backend añadirá al final tarjetas HTML con botones "Agregar al carrito" y "Ver producto".
- Usa el contexto previo para entender referencias ("sí", "no", "esa", "el primero", etc.) y mantén coherencia dentro de la sesión.
"""

# =============================
# Rutas
# =============================
@app.get("/")
def root():
    return {
        "service": "EcoAmigo API",
        "version": app.version,
        "allowed_origins": origins,
        "catalog_size": len(CATALOG),
        "memory_sessions": len(MEMORY),
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
async def chat_endpoint(payload: ChatPayload, request: Request):
    user_msg = payload.message or payload.prompt or payload.query or payload.input_text or ""
    session_id = payload.session_id or "no-session"

    if not user_msg.strip():
        return {"answer": "¿Me cuentas un poco más? ¿Qué necesitas sobre sustentabilidad o productos del hogar?"}

    # 1) Decide si vamos a recomendar (según intención)
    history = MEMORY[session_id]
    recommend_now = should_recommend(user_msg, history)

    # 2) Si vamos a recomendar, calcula matches y tarjetas; si no, deja vacío
    matches: List[Dict[str, Any]] = match_products(user_msg, topk=3) if recommend_now else []
    html_block = render_product_cards(matches) if matches else ""

    # 3) Construir mensajes con MEMORIA por sesión
    messages_for_model = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages_for_model.extend(list(history))
    messages_for_model.append({"role": "user", "content": user_msg})

    # 4) Llamar modelo (o fallback)
    client = build_client()
    if client:
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages_for_model,
                temperature=0.4,
            )
            base_answer = (resp.choices[0].message.content or "").strip()
        except Exception:
            base_answer = (
                "Puedo ayudarte con consejos de vida sustentable y, si lo deseas, puedo recomendarte productos adecuados."
            )
    else:
        base_answer = (
            "Este es un modo demostración sin conexión a OpenAI. Puedo orientarte y, si lo deseas, sugerir productos."
        )

    # 5) Limpieza para evitar duplicados de precio/enlaces/HTML en el texto libre
    if matches and base_answer:
        base_answer = _re.sub(r"```[\s\S]*?```", "", base_answer)
        base_answer = _re.sub(r"(?im)^\s*\**precio\**\s*:.*$", "", base_answer)
        urls = [p.get("url", "") for p in matches if p.get("url")]
        for u in urls:
            if u:
                base_answer = _re.sub(rf"\[([^\]]+)\]\({_re.escape(u)}\)", r"\1", base_answer)
        base_answer = _re.sub(r"\n{3,}", "\n\n", base_answer).strip()

    # 6) Respuesta final (si recommend_now → texto + tarjetas; si no → solo texto)
    final_answer = base_answer + (f"\n\n{html_block}" if recommend_now and html_block else "")

    # 7) Actualizar memoria
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": base_answer})

    # 8) Log
    log_chat(session_id, user_msg, final_answer)

    return {"answer": final_answer}

# =============================
# Arranque local opcional
# =============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
