# main.py — EcoAmigo Chat API (v1.7)
# Mejoras clave:
# - Carga robusta de catálogo (corrige catalog_size:0 en Railway/local)
# - Normalización acento‑insensible y sinónimos básicos ES para mejor matching
# - Fallback para SIEMPRE mostrar tarjetas (si no hay match, muestra top N)
# - Prompt ajustado: el LLM NO genera HTML de productos (lo hace el backend)
# - CORS saneado: agrega https:// a orígenes sin esquema
# - Endpoints de debug: /catalog y /admin/reload (opcional con ADMIN_TOKEN)

from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from collections import deque
from pathlib import Path
import os, json, re, csv, datetime, logging, unicodedata

load_dotenv()

APP_VERSION = "1.7.0"
app = FastAPI(title="EcoAmigo Chat API", version=APP_VERSION)

# ------------------------- CORS -------------------------
_raw = os.getenv("ALLOWED_ORIGINS", "")
_raw_list = [o.strip() for o in _raw.split(",") if o.strip()]

def _sanitize_origin(o: str) -> Optional[str]:
    if not o:
        return None
    if o.startswith("http://") or o.startswith("https://"):
        return o
    # Si viene sin esquema, asumimos https
    return f"https://{o}"

ALLOWED_ORIGINS = [o for o in (_sanitize_origin(x) for x in _raw_list) if o]
# Valores por defecto (útil en dev si .env no define ALLOWED_ORIGINS)
if not ALLOWED_ORIGINS:
    ALLOWED_ORIGINS = [
        "https://grinhaus.co",
        "https://a71669-e2.myshopify.com",
        "https://TU-NGROK.ngrok-free.app",
        "https://web-production-ea82.up.railway.app",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# --------------------- Logging Requests ---------------------
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

# ------------------------- OpenAI -------------------------
try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

# ------------------------- Catálogo -------------------------
BASE_DIR = Path(__file__).resolve().parent
CATALOG_PATH = os.getenv("CATALOG_PATH", "products.json")

# lista en memoria
CATALOG: List[Dict[str, Any]] = []


def _load_catalog_candidates() -> List[str]:
    """Posibles rutas para el catálogo (hace robusta la carga en Railway)."""
    candidates = [
        CATALOG_PATH,
        str(BASE_DIR / CATALOG_PATH),
        str(BASE_DIR / "products.json"),
        str(Path.cwd() / CATALOG_PATH),
    ]
    # quitar duplicados preservando orden
    seen = set()
    out = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def load_catalog() -> List[Dict[str, Any]]:
    for path in _load_catalog_candidates():
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        logger.info(f"[catalog] cargado: {path} ({len(data)} items)")
                        return data
        except Exception as e:
            logger.warning(f"[catalog] error leyendo {path}: {e}")
    logger.warning("[catalog] no se encontró catálogo. Usando lista vacía.")
    return []


CATALOG = load_catalog()

# ------------------ Logging a CSV ------------------
LOG_DIR = os.getenv("LOG_DIR", "logs")
LOG_FILE = os.path.join(LOG_DIR, "chat_logs.csv")
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "timestamp", "ip", "session_id", "user_text", "answer", "matched_products"
        ])


def log_chat(ip: str, sid: str, user_text: str, answer: str, matched_products: List[str]):
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            datetime.datetime.utcnow().isoformat(), ip, sid, user_text, answer, "|".join(matched_products)
        ])

# ------------------ Personalización de tono ------------------
SHOP_NAME = os.getenv("SHOP_NAME", "GrinHaus")
SHOP_TONE = os.getenv(
    "SHOP_TONE",
    "amable, cercano, experto en vida sustentable y productos ecológicos del hogar",
)

SYSTEM_PROMPT = f"""
Eres EcoAmigo, asistente de {SHOP_NAME}. Tu tono es {SHOP_TONE}.
- Mantén el contexto de la conversación.
- Responde claro y útil. Si corresponde, sugiere referencias a productos del catálogo local ya proporcionado en el *contexto* del mensaje.
- No inventes datos; usa sólo la información del catálogo incluida en el contexto.
- MUY IMPORTANTE: No generes HTML de tarjetas ni botones. El backend añadirá las tarjetas de productos automáticamente.
- Si el usuario responde con "sí", "no", "recomiéndame", etc., usa el contexto previo para entender la referencia.
"""

# ------------------ Matching mejorado ------------------
_re_ws = re.compile(r"[\W_]+", flags=re.UNICODE)


def _strip_accents(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")


def normalize(s: Optional[str]) -> str:
    s = s or ""
    s = _strip_accents(s.lower())
    s = _re_ws.sub(" ", s).strip()
    return s


# Sinónimos y ampliadores simples en ES
SYNONYMS: Dict[str, List[str]] = {
    "detergente": ["detergente", "lavar", "lavado", "ropa", "manchas", "suavizante"],
    "esponja": ["esponja", "lavaloza", "platos", "cocina", "fibra", "lufa"],
    "cepillo": ["cepillo", "dientes", "bambu", "higiene", "oral"],
    "baño": ["bano", "ducha", "jabon", "shampoo", "aseo"],
    "cocina": ["cocina", "platos", "ollas", "sarten", "lavaloza"],
}


def _expand_words(words: List[str]) -> List[str]:
    exp = set(words)
    for w in list(words):
        for _, vals in SYNONYMS.items():
            if w in vals:
                exp.update(vals)
    return list(exp)


def _product_haystack(p: Dict[str, Any]) -> Dict[str, str]:
    title = normalize(p.get("title", ""))
    tags = normalize(" ".join(p.get("tags", []) or []))
    desc = normalize(p.get("description", ""))
    cat = normalize(p.get("category", ""))
    return {
        "title": title,
        "all": " ".join(filter(None, [title, tags, desc, cat])).strip(),
    }


def find_products(query: str, k: int = 3) -> List[Dict[str, Any]]:
    q = normalize(query)
    if not q or not CATALOG:
        return []
    words = _expand_words(q.split())

    scored: List[tuple[int, Dict[str, Any]]] = []
    for p in CATALOG:
        hay = _product_haystack(p)
        score = 0
        for w in words:
            if w and w in hay["all"]:
                score += 1
                if w in hay["title"]:
                    score += 2  # bonus por aparecer en el título
        if score > 0:
            scored.append((score, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [p for _, p in scored[:k]]

    # Fallback: si no hay match, devolvemos los primeros k del catálogo para asegurar tarjetas
    if not top:
        top = CATALOG[:k]
    return top

# ------------------ Memoria por sesión (RAM) ------------------
SESSIONS: Dict[str, deque] = {}
MAX_HISTORY = 8  # últimos 8 turnos (user+assistant = 16 mensajes)


def get_history(sid: str) -> deque:
    if sid not in SESSIONS:
        SESSIONS[sid] = deque(maxlen=MAX_HISTORY * 2)
    return SESSIONS[sid]

# ------------------ Pydantic ------------------
class ChatRequest(BaseModel):
    message: Optional[str] = None
    prompt: Optional[str] = None
    query: Optional[str] = None
    input_text: Optional[str] = None
    session_id: Optional[str] = None

    def text(self) -> str:
        for v in (self.message, self.prompt, self.query, self.input_text):
            if v and v.strip():
                return v.strip()
        raise ValueError("Falta texto (message/prompt/query/input_text).")

# ------------------ Utilidades de tarjetas ------------------

def _build_cards_html(matches: List[Dict[str, Any]]) -> str:
    cards: List[str] = []
    for p in matches:
        title = p.get("title", "Producto")
        url = p.get("url", "#")
        vid = str(p.get("variant_id", "")).strip()
        price = p.get("price", "")
        price_html = f"<div style='color:#065f46'>Precio: {price}</div>" if price else ""

        if vid:
            add_btn = (
                f"<button data-variant-id=\"{vid}\" "
                "style=\"padding:6px 10px; border-radius:8px; background:#16a34a; "
                "color:#fff; border:none; cursor:pointer\">Agregar al carrito</button>"
            )
        else:
            add_btn = ""

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
    return "\n".join(cards)

# ------------------ Rutas ------------------
@app.get("/")
def root():
    return {
        "service": "EcoAmigo Chat API",
        "version": APP_VERSION,
        "allowed_origins": ALLOWED_ORIGINS,
        "catalog_size": len(CATALOG),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/catalog")
def catalog_info():
    return {"catalog_size": len(CATALOG)}


ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "").strip()


@app.post("/admin/reload")
def admin_reload(token: Optional[str] = Body(default=None)):
    if ADMIN_TOKEN and token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Token inválido")
    global CATALOG
    CATALOG = load_catalog()
    return {"catalog_size": len(CATALOG)}


@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    try:
        user_text = req.text()
        sid = (req.session_id or "").strip() or "anon"
        ip = request.client.host if request and request.client else ""

        # 1) Productos (para enriquecer contexto + generar tarjetas)
        matches = find_products(user_text, k=3)
        matched_ids: List[str] = [str(p.get("variant_id", "")).strip() for p in matches if p.get("variant_id")]

        # Construir contexto de productos para el LLM (sin HTML)
        products_context = ""
        if matches:
            lines = []
            for p in matches:
                lines.append(
                    f"- {p.get('title')} | precio: {p.get('price','')} | url: {p.get('url','')} | variant_id: {p.get('variant_id','')}"
                )
            products_context = "Productos relevantes:\n" + "\n".join(lines)

        # 2) Historial + mensaje actual
        history = get_history(sid)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(list(history))
        user_prompt = user_text + (f"\n\n{products_context}\n\n" if products_context else "")
        messages.append({"role": "user", "content": user_prompt})

        # 3) Llamar a OpenAI (o demo)
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not _HAS_OPENAI or not api_key:
            base_answer = f"(Demo sin OpenAI) Recibí: {user_text}"
        else:
            client = OpenAI()
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                temperature=0.6,
            )
            base_answer = (
                resp.choices[0].message.content if resp and getattr(resp, "choices", None) else "No pude generar respuesta."
            )

        # 4) Tarjetas HTML (siempre mostramos algo: matches o fallback ya forzado arriba)
        html_block = _build_cards_html(matches)

        final_answer = base_answer + (f"\n\n{html_block}" if html_block else "")

        # 5) Actualizar historial (user + assistant)
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": base_answer})

        # 6) Log
        log_chat(ip, sid, user_text, final_answer, matched_ids)

        return {"answer": final_answer, "session_id": sid}

    except Exception as e:
        msg = str(e)
        if "quota" in msg.lower() or "429" in msg:
            return {"answer": "⚠️ Sin crédito suficiente en OpenAI ahora mismo. (Respuesta de prueba) 🌱"}
        raise HTTPException(status_code=500, detail=f"Error interno: {msg}")
