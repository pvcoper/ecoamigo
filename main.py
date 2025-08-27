import os
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import httpx
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
app = FastAPI(title="EcoAmigo API", version="1.7")

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
    Búsqueda muy simple por palabras clave sobre title/tags.
    Puedes mejorar a TF-IDF/embeddings cuando quieras.
    """
    if not user_text or not CATALOG:
        return []
    q = user_text.lower()
    scored = []
    for p in CATALOG:
        title = (p.get("title") or "").lower()
        tags = " ".join(p.get("tags", [])).lower() if isinstance(p.get("tags"), list) else str(p.get("tags") or "").lower()
        score = 0
        # Suma puntos por apariciones
        for token in set(q.split()):
            if token and (token in title or token in tags):
                score += 1
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
# Prompt del sistema (con regla nueva)
# =============================
SYSTEM_PROMPT = f"""
Eres EcoAmigo, asistente de {SHOP_NAME}. Tu tono es {SHOP_TONE}.
- Mantén el contexto de la conversación.
- Responde claro y útil. Si corresponde, sugiere productos del catálogo local.
- No inventes datos; usa solo el catálogo entregado por el backend.
- Al recomendar, ofrece un bloque HTML con botones "Agregar al carrito" (data-variant-id) y "Ver producto".
- Si el usuario responde con "sí", "no", "recomiéndame", etc., usa el contexto previo para entender la referencia.

- MUY IMPORTANTE: En el TEXTO LIBRE de tu respuesta NO incluyas precios, enlaces ni bloques HTML de productos.
  Deja esas acciones solo en las TARJETAS HTML que agrega el backend.
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

    # 1) Matches del catálogo
    matches = match_products(user_msg, topk=3)
    html_block = render_product_cards(matches) if matches else ""

    # 2) Construir respuesta base (modelo o fallback)
    client = build_client()
    if client:
        # ----- OpenAI -----
        msgs = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_msg},
        ]
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=msgs,
                temperature=0.4,
            )
            base_answer = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            base_answer = (
                "Puedo recomendarte alternativas sustentables y, al final, verás botones para agregarlas al carrito o abrir el producto."
            )
    else:
        # ----- Fallback sin OpenAI -----
        base_answer = (
            "Este es un modo demostración sin conexión a OpenAI. A continuación, verás sugerencias con botones para agregar al carrito o ver producto."
        )

    # 3) Limpieza para evitar duplicados (precio/enlaces/html en el texto libre)
    if matches and base_answer:
        import re as _re
        # Quitar bloques de código ``` ... ```
        base_answer = _re.sub(r"```[\s\S]*?```", "", base_answer)
        # Quitar líneas "Precio:" / "**Precio**:"
        base_answer = _re.sub(r"(?im)^\s*\**precio\**\s*:.*$", "", base_answer)
        # Quitar enlaces markdown hacia URLs del catálogo
        urls = [p.get("url", "") for p in matches if p.get("url")]
        for u in urls:
            if not u:
                continue
            base_answer = _re.sub(rf"\[([^\]]+)\]\({_re.escape(u)}\)", r"\1", base_answer)
        # Compactar saltos en blanco
        base_answer = _re.sub(r"\n{3,}", "\n\n", base_answer).strip()

    # 4) Respuesta final (texto + tarjetas)
    final_answer = base_answer + (f"\n\n{html_block}" if html_block else "")

    # 5) Log
    log_chat(session_id, user_msg, final_answer)

    return {"answer": final_answer}


# =============================
# Arranque local opcional
# =============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
