# EcoAmigo Chat API - v2.1 (intención + contexto de catálogo + memoria + exclusiones)
import os, json, csv, pathlib, datetime, logging, re as _re
from typing import Optional, List, Dict, Any
from collections import deque, defaultdict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# -------------------------------------------------
# App & CORS
# -------------------------------------------------
app = FastAPI(title="EcoAmigo Chat API", version="2.1")

_raw = os.getenv("ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS = [o.strip() for o in _raw.split(",") if o.strip()]
if not ALLOWED_ORIGINS:
    # defaults seguros (ajusta con tu dominio)
    ALLOWED_ORIGINS = ["https://grinhaus.co", "https://a71669-e2.myshopify.com"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Logging simple de requests
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

# -------------------------------------------------
# OpenAI
# -------------------------------------------------
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# -------------------------------------------------
# Catálogo
# -------------------------------------------------
CATALOG_PATH = os.getenv("CATALOG_PATH", "products.json")

REQUIRED_FIELDS = {"title", "url", "variant_id"}  # mínimo para tarjetas

def _is_valid_product(p: Dict[str, Any]) -> bool:
    if not isinstance(p, dict): return False
    if not REQUIRED_FIELDS.issubset(p.keys()): return False
    if not str(p.get("title", "")).strip(): return False
    if not str(p.get("url", "")).strip(): return False
    if not str(p.get("variant_id", "")).strip(): return False
    return True

def load_catalog(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path): return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list): return []
            cleaned = [p for p in data if _is_valid_product(p)]
            return cleaned
    except Exception:
        return []

CATALOG = load_catalog(CATALOG_PATH)

# -------------------------------------------------
# Logs a CSV (como v1.6)
# -------------------------------------------------
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

# -------------------------------------------------
# Personalización
# -------------------------------------------------
SHOP_NAME = os.getenv("SHOP_NAME", "GrinHaus")
SHOP_TONE = os.getenv("SHOP_TONE", "amable, cercano, experto en vida sustentable y productos ecológicos del hogar")

SYSTEM_PROMPT = f"""
Eres EcoAmigo, asistente de {SHOP_NAME}. Tu tono es {SHOP_TONE}.
- Mantén el contexto de la conversación.
- Eres experto en sustentabilidad para el hogar; si te preguntan algo fuera, redirige suavemente.
- Responde claro y útil.
- Muy importante: solo recomienda productos si el usuario lo pide explícitamente o si el contexto lo amerita.
  Si el usuario solo saluda o se presenta, responde cordial y pregunta qué necesita antes de sugerir productos.
- Cuando recomiendes, NO incluyas precios, enlaces ni HTML en el texto libre. El backend añadirá tarjetas con botones.
- Cuando nombres un producto, usa únicamente los títulos que existen en el catálogo disponible (no inventes nombres ni enlaces).
- Si no hay coincidencias en el catálogo para lo que pide el usuario, dilo con claridad y ofrece alternativas generales sin mencionar marcas.
- Si el usuario pide "links", "acceso directo", "ver producto", etc., explícales brevemente que verá los enlaces en las tarjetas al final.
"""

# -------------------------------------------------
# Utilidades de matching
# -------------------------------------------------
def normalize(s: str) -> str:
    # igual a v1.6, pero conservamos minúsculas y quitamos ruido
    return _re.sub(r"[\W_]+", " ", (s or "").lower()).strip()

def match_products(user_text: str, topk: int = 3) -> List[Dict[str, Any]]:
    """
    Busca SOLO en CATALOG (ya validado) y descarta por exclude_keywords:
      - Palabra suelta => \b...\b
      - Frase => subcadena
    Scoring simple por coincidencias en título/tags.
    """
    if not user_text or not CATALOG: return []
    q = user_text.lower()
    qs = set(q.split())

    def is_excluded(prod: Dict[str, Any]) -> bool:
        ex = prod.get("exclude_keywords") or []
        for kw in ex:
            if not kw: continue
            kw_norm = str(kw).lower().strip()
            if not kw_norm: continue
            if " " in kw_norm:
                if kw_norm in q:  # frase
                    return True
            else:
                if _re.search(rf"\b{_re.escape(kw_norm)}\b", q):  # palabra
                    return True
        return False

    scored = []
    for p in CATALOG:
        if not _is_valid_product(p): 
            continue
        if is_excluded(p):
            continue

        title = normalize(p.get("title",""))
        tags  = " ".join([normalize(t) for t in (p.get("tags") or [])])
        hay   = f"{title} {tags}"
        score = sum(1 for w in qs if w and w in hay)
        if score > 0:
            scored.append((score, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:topk]]

def products_context_lines(products: List[Dict[str, Any]]) -> str:
    """
    Devuelve líneas con info del catálogo para el modelo (sin URL para no tentar a poner links).
    """
    if not products: return ""
    lines = []
    for p in products:
        title = p.get("title","")
        price = p.get("price","")
        tags  = ", ".join(p.get("tags",[]) or [])
        # evitamos URL acá (las tarjetas ya la incluyen)
        lines.append(f"- {title} | precio: {price} | tags: {tags}")
    return "Productos relevantes en catálogo:\n" + "\n".join(lines)

# -------------------------------------------------
# Intención (matizada) + memoria
# -------------------------------------------------
YES_WORDS = {"si","sí","dale","ok","okey","de acuerdo","perfecto","claro","bueno","vale","ya","hagámoslo","hágamoslo"}

GREETING_PATTERNS = _re.compile(
    r"\b(hola|buenas|buenos\s+d[ií]as|buenas\s+tardes|buenas\s+noches|qué\s+tal|como\s+est[aá]s|soy\s+\w+)\b",
    flags=_re.IGNORECASE
)

EXPLICIT_RECO_PATTERNS = _re.compile(
    r"(recom(i|e)nd(a|ame|ar|aciones?)|¿\s*qu[eé]\s+me\s+recomiendas|puedes\s+recomendar(me)?|"
    r"me\s+recom(i|e)endas|sug(e|i)r(e|e)?(me)?|suger(e|i)ncia(s)?|puedes\s+sugerir(me)?|me\s+sugieres|"
    r"opciones?|alternativa(s)?|quiero\s+(ver\s+)?(opciones|productos?|comprar)|"
    r"busco|estoy\s+buscando|ando\s+necesitando|necesito|"
    r"qué\s+opciones|qué\s+alternativas|qué\s+productos|qué\s+hay\s+para|"
    r"tienes\s+algo|qué\s+tienen|muestran\s+productos|"
    r"dame\s+un\s+dato|alg[uú]n\s+dato|datito(s)?|dame\s+un\s+tip|alg[uú]n\s+tip|"
    r"dame\s+una\s+idea|consejito(s)?)",
    flags=_re.IGNORECASE
)

PURCHASE_INTENT_PATTERNS = _re.compile(
    r"\b(comprar|compra|precio|cu[eé]sta|vale|agregar|a[nñ]adir|carrito|ver\s+producto|cotizar|"
    r"enlace|link|url|acceso\s+directo|p[aá]same\s+el\s+link|p[aá]same\s+los\s+links)\b",
    flags=_re.IGNORECASE
)

def said_yes(t: str) -> bool:
    t = " " + (t or "").lower().strip() + " "
    return any((" " + w + " ") in t for w in YES_WORDS)

def is_greeting_only(t: str) -> bool:
    t = (t or "").strip()
    return (len(t) <= 25) and bool(GREETING_PATTERNS.search(t))

def has_intent_now(t: str) -> bool:
    if is_greeting_only(t): return False
    if EXPLICIT_RECO_PATTERNS.search(t): return True
    if PURCHASE_INTENT_PATTERNS.search(t): return True
    return False

def assistant_offered_recommendation(text: str) -> bool:
    if not text: return False
    t = text.lower()
    return ("¿quieres que te recomiende" in t or
            "¿te sugiero" in t or
            "¿quieres que te muestre" in t or
            "¿quieres recomendaciones" in t)

def should_recommend(user_msg: str, history: deque) -> bool:
    if has_intent_now(user_msg): return True
    if history:
        last_assistant = None
        for m in reversed(history):
            if m["role"] == "assistant":
                last_assistant = m["content"]; break
        if last_assistant and assistant_offered_recommendation(last_assistant):
            return said_yes(user_msg)
    return False

# Memoria de diálogo y de últimos matches (para “esos productos/links”)
SESSIONS: Dict[str, deque] = {}                         # sid -> deque de mensajes
MAX_HISTORY = 8
LAST_MATCHES: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

def get_history(sid: str) -> deque:
    if sid not in SESSIONS:
        SESSIONS[sid] = deque(maxlen=MAX_HISTORY*2)
    return SESSIONS[sid]

# -------------------------------------------------
# Pydantic
# -------------------------------------------------
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

# -------------------------------------------------
# Rutas
# -------------------------------------------------
@app.get("/")
def root():
    return {
        "service": "EcoAmigo Chat API",
        "version": app.version,
        "allowed_origins": ALLOWED_ORIGINS,
        "catalog_size": len(CATALOG),
        "memory_sessions": len(SESSIONS),
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    try:
        user_text = req.text()
        sid = (req.session_id or "").strip() or "anon"
        ip = request.client.host if request and request.client else ""

        # 1) Siempre buscamos productos para enriquecer el contexto del modelo
        matches_now = match_products(user_text, topk=3)
        products_ctx = products_context_lines(matches_now)

        # 2) Intención de tarjetas
        history = get_history(sid)
        recommend_now = should_recommend(user_text, history)

        # 3) Armar mensajes al modelo (system + history + turno + contexto S/URL)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(list(history))
        user_prompt = user_text + (f"\n\n{products_ctx}\n\n" if products_ctx else "")
        messages.append({"role": "user", "content": user_prompt})

        # 4) Llamar a OpenAI (o demo)
        if not OPENAI_API_KEY:
            base_answer = f"(Demo sin OpenAI) Recibí: {user_text}"
        else:
            client = OpenAI(api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                temperature=0.5,
            )
            base_answer = resp.choices[0].message.content if resp and resp.choices else "No pude generar respuesta."

        # 5) Decidir tarjetas (con fallback a últimos matches si piden links)
        matches_for_cards: List[Dict[str, Any]] = []
        if recommend_now:
            matches_for_cards = matches_now or LAST_MATCHES.get(sid, [])
        html_block = ""
        matched_ids: List[str] = []
        if matches_for_cards:
            cards = []
            for p in matches_for_cards:
                if not _is_valid_product(p): 
                    continue
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

        # 6) Limpieza anti-duplicados en texto (sin precios/enlaces/html)
        if matches_for_cards and base_answer:
            base_answer = _re.sub(r"```[\s\S]*?```", "", base_answer)
            base_answer = _re.sub(r"(?im)^\s*\**precio\**\s*:.*$", "", base_answer)
            # también removemos menciones explícitas de URL si las hubiera
            base_answer = _re.sub(r"https?://\S+", "", base_answer).strip()
            base_answer = _re.sub(r"\n{3,}", "\n\n", base_answer).strip()

        # 7) Si hubo intención pero sin matches: mensaje claro
        if recommend_now and not matches_for_cards:
            base_answer = (base_answer + "\n\n"
                           "Por ahora no tengo productos de nuestro catálogo que calcen con eso. "
                           "Si quieres, puedo sugerirte pautas generales o buscar alternativas cercanas."
                           ).strip()

        # 8) Respuesta final
        final_answer = base_answer + (f"\n\n{html_block}" if recommend_now and html_block else "")

        # 9) Actualizar historial + últimos matches
        history.append({"role":"user","content": user_text})
        history.append({"role":"assistant","content": base_answer})
        if matches_now:
            LAST_MATCHES[sid] = matches_now

        # 10) Log
        log_chat(ip, sid, user_text, final_answer, matched_ids)

        return {"answer": final_answer, "session_id": sid}

    except Exception as e:
        msg = str(e)
        if "quota" in msg.lower() or "429" in msg:
            return {"answer":"⚠️ Sin crédito suficiente en OpenAI ahora mismo. (Respuesta de prueba) 🌱"}
        raise HTTPException(status_code=500, detail=f"Error interno: {msg}")
