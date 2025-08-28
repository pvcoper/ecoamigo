# main.py (v1.6) - memoria por sesión
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from collections import deque
import os, json, re, csv, pathlib, datetime, logging

load_dotenv()

app = FastAPI(title="EcoAmigo Chat API", version="1.6.0")

# ---- CORS ----
_raw = os.getenv("ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS = [o.strip() for o in _raw.split(",") if o.strip()]
if not ALLOWED_ORIGINS:
    ALLOWED_ORIGINS = [
        "https://grinhaus.co",
        "https://a71669-e2.myshopify.com",
        "https://TU-NGROK.ngrok-free.app",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ---- Logging simple de requests ----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ecoamigo")

@app.middleware("http")
async def log_requests(request, call_next):
    client = request.client.host if request.client else "unknown"
    logger.info(f"--> {client} {request.method} {request.url.path}")
    try:
        response = await call_next(request)
        logger.info(f"<-- {client} {request.method} {request.url.path} {response.status_code}")
        return response
    except Exception as e:
        logger.exception(f"xx> {client} {request.method} {request.url.path} ERROR: {e}")
        raise

# ---- OpenAI ----
try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

# ---- Catálogo ----
CATALOG_PATH = os.getenv("CATALOG_PATH", "products.json")
def load_catalog(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path): return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data if isinstance(data, list) else []
CATALOG = load_catalog(CATALOG_PATH)

# ---- Logging a CSV ----
LOG_DIR = os.getenv("LOG_DIR", "logs")
LOG_FILE = os.path.join(LOG_DIR, "chat_logs.csv")
pathlib.Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["timestamp","ip","session_id","user_text","answer","matched_products"])
def log_chat(ip: str, sid: str, user_text: str, answer: str, matched_products: List[str]):
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            datetime.datetime.utcnow().isoformat(), ip, sid, user_text, answer, "|".join(matched_products)
        ])

# ---- Personalización de tono ----
SHOP_NAME = os.getenv("SHOP_NAME", "GrinHaus")
SHOP_TONE = os.getenv("SHOP_TONE", "amable, cercano, experto en vida sustentable y productos ecológicos del hogar")

SYSTEM_PROMPT = f"""
Eres EcoAmigo, asistente de {SHOP_NAME}. Tu tono es {SHOP_TONE}.
- Mantén el contexto de la conversación.
- Responde claro y útil. Si corresponde, sugiere productos del catálogo local.
- No inventes datos; usa solo el catálogo entregado por el backend.
- Al recomendar, ofrece un bloque HTML con botones "Agregar al carrito" (data-variant-id) y "Ver producto".
- Si el usuario responde con "sí", "no", "recomiéndame", etc., usa el contexto previo para entender la referencia.
"""

# ---- Matching simple ----
import re as _re
def normalize(s: str) -> str:
    return _re.sub(r"[\W_]+", " ", (s or "").lower()).strip()

def find_products(query: str, k: int = 3) -> List[Dict[str, Any]]:
    q = normalize(query)
    if not q or not CATALOG: return []
    qs = set(q.split())
    scored = []
    for p in CATALOG:
        hay = normalize(p.get("title","")) + " " + " ".join([normalize(t) for t in p.get("tags",[])])
        score = sum(1 for w in qs if w in hay)
        if score > 0: scored.append((score, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:k]]

# ---- Memoria por sesión (en RAM) ----
# SESSIONS[sid] = deque([{"role":"user"/"assistant","content": "..."}], maxlen=MAX_HISTORY*2)
SESSIONS: Dict[str, deque] = {}
MAX_HISTORY = 8  # últimos 8 turnos (user+assistant = 16 mensajes)

def get_history(sid: str) -> deque:
    if sid not in SESSIONS:
        SESSIONS[sid] = deque(maxlen=MAX_HISTORY*2)
    return SESSIONS[sid]

# ---- Pydantic ----
class ChatRequest(BaseModel):
    message: Optional[str] = None
    prompt: Optional[str] = None
    query: Optional[str] = None
    input_text: Optional[str] = None
    session_id: Optional[str] = None

    def text(self) -> str:
        for v in (self.message, self.prompt, self.query, self.input_text):
            if v and v.strip(): return v.strip()
        raise ValueError("Falta texto (message/prompt/query/input_text).")

# ---- Rutas ----
@app.get("/")
def root():
    return {"service":"EcoAmigo Chat API","version":"1.6.0","allowed_origins":ALLOWED_ORIGINS,"catalog_size":len(CATALOG)}

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    try:
        user_text = req.text()
        sid = (req.session_id or "").strip() or "anon"
        ip = request.client.host if request and request.client else ""

        # 1) Buscar productos según el último input (para enriquecer el contexto)
        matches = find_products(user_text, k=3)
        products_context = ""
        if matches:
            lines = []
            for p in matches:
                lines.append(f"- {p.get('title')} | precio: {p.get('price','')} | url: {p.get('url','')} | variant_id: {p.get('variant_id','')}")
            products_context = "Productos relevantes:\n" + "\n".join(lines)

        # 2) Construir historial + mensaje actual
        history = get_history(sid)
        messages = [{"role":"system","content": SYSTEM_PROMPT}]
        # añade historial previo
        messages.extend(list(history))
        # añade turno actual (con contexto de productos si lo hay)
        user_prompt = user_text + (f"\n\n{products_context}\n\n" if products_context else "")
        messages.append({"role":"user","content": user_prompt})

        # 3) Llamar a OpenAI (o demo)
        api_key = os.getenv("OPENAI_API_KEY","").strip()
        if not _HAS_OPENAI or not api_key:
            base_answer = f"(Demo sin OpenAI) Recibí: {user_text}"
        else:
            client = OpenAI()
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                temperature=0.6,
            )
            base_answer = resp.choices[0].message.content if resp and resp.choices else "No pude generar respuesta."

        # 4) Tarjetas HTML si hubo matches
        html_block = ""
        matched_ids: List[str] = []
        if matches:
            cards = []
            for p in matches:
                title = p.get("title", "Producto")
                url = p.get("url", "#")
                vid = str(p.get("variant_id", "")).strip()
                price = p.get("price", "")
                if vid:
                    matched_ids.append(vid)
                    add_btn = (
                        f'<button data-variant-id="{vid}" '
                        'style="padding:6px 10px; border-radius:8px; background:#16a34a; '
                        'color:#fff; border:none; cursor:pointer">Agregar al carrito</button>'
                    )
                else:
                    add_btn = ""
                price_html = f"<div style='color:#065f46'>Precio: {price}</div>" if price else ""
                card = (
                    "<div style='margin:10px 0; padding:10px; background:#f6fef7; "
                    "border:1px solid #d1fadf; border-radius:10px'>"
                    f"<div style='font-weight:600'>{title}</div>"
                    f"{price_html}"
                    "<div style='margin-top:6px; display:flex; gap:8px; flex-wrap:wrap'>"
                    f"{add_btn}"
                    f"<a href='{url}' target='_blank' "
                    "style='padding:6px 10px; border-radius:8px; background:#e5f3ff; "
                    "color:#0b4a6e; text-decoration:none'>Ver producto</a>"
                    "</div>"
                    "</div>"
                )
                cards.append(card)
            html_block = "\n".join(cards)

        final_answer = base_answer + (f"\n\n{html_block}" if html_block else "")

        # 5) Actualizar historial (user + assistant)
        history.append({"role":"user","content": user_text})
        history.append({"role":"assistant","content": base_answer})

        # 6) Log
        log_chat(ip, sid, user_text, final_answer, matched_ids)

        return {"answer": final_answer, "session_id": sid}

    except Exception as e:
        msg = str(e)
        if "quota" in msg.lower() or "429" in msg:
            return {"answer":"⚠️ Sin crédito suficiente en OpenAI ahora mismo. (Respuesta de prueba) 🌱"}
        raise HTTPException(status_code=500, detail=f"Error interno: {msg}")
