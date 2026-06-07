

LEIA_CSS = """
<style>
/* ─── Google Fonts ─────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;1,300&family=Jost:wght@300;400;500&display=swap');

/* ─── Variables LÉIA ───────────────────────────────── */
:root {
    --leia-grey:        #FAE7D7;   /* fond principal */
    --leia-terracotta:  #DDB99A;   /* fond sidebar */
    --leia-surface:     #E8CFBA;   /* surface bulles/cards */
    --leia-border:      #C0C0C0;   /* bordures */
    --leia-gold:        #261F1C;   /* accent hover, focus */
    --leia-ivory:       #261F1C;   /* texte principal */
    --leia-muted:       #1C1411;   /* texte secondaire */
}

/* ─── App background ───────────────────────────────── */
.stApp {
    background-color: var(--leia-grey) !important;
    font-family: 'Jost', sans-serif !important;
}

/* ─── Main content area ────────────────────────────── */
.main .block-container {
    background-color: var(--leia-grey) !important;
    padding-top: 2rem;
    max-width: 820px;
}

[data-testid="stBottomBlockContainer"] {
    background-color: var(--leia-grey) !important;
}

[data-testid="stHeader"] {
    background-color: var(--leia-grey) !important;
}

/* ─── Title ────────────────────────────────────────── */
h1 {
    font-family: 'Cormorant Garamond', serif !important;
    font-weight: 300 !important;
    font-size: 2.2rem !important;
    color: var(--leia-gold) !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid var(--leia-border);
    padding-bottom: 0.6rem;
    margin-bottom: 0.2rem;
}

[data-testid="stSidebar"] h1 {
    border-bottom: none !important;
}

/* ─── Subtitle / description ───────────────────────── */
p, .stMarkdown p {
    font-family: 'Jost', sans-serif !important;
    color: var(--leia-muted) !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.04em;
}

/* ─── Sidebar ──────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: var(--leia-terracotta) !important;
    border-right: 1px solid var(--leia-border) !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: 'Cormorant Garamond', serif !important;
    color: var(--leia-gold) !important;
    font-weight: 300 !important;
    letter-spacing: 0.1em;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: var(--leia-muted) !important;
    font-family: 'Jost', sans-serif !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.03em;
}

/* ─── Select your role box────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
    background-color: var(--leia-surface) !important;
    border: 1px solid var(--leia-ivory) !important;
    color: var(--leia-ivory) !important;
    font-family: 'Jost', sans-serif !important;
    font-size: 0.85rem !important;
    border-radius: 2px !important;
}

[data-testid="stSelectbox"] > div > div:hover {
    border-color: var(--leia-gold) !important;
}

/* ─── Chat input ───────────────────────────────────── */
[data-testid="stChatInput"] textarea {
    background-color: #FFFAFA !important;
    border: 1px solid var(--leia-border) !important;
    color: var(--leia-muted) !important;
    font-family: 'Jost', sans-serif !important;
    font-size: 0.9rem !important;
    border-radius: 2px !important;
    caret-color: var(--leia-muted);
}

[data-testid="stChatInput"] textarea:focus {
    border-color: var(--leia-gold) !important;
    box-shadow: 0 0 0 1px var(--leia-gold) !important;
    outline: none !important;
}

[data-testid="stChatInput"] > div:focus-within {
    border-color: var(--leia-gold) !important;
    box-shadow: 0 0 0 1px var(--leia-gold) !important;
    outline: none !important;
}

/* ─── Chat messages — user ─────────────────────────── */
[data-testid="stChatMessage"][data-testid*="user"],
.stChatMessage:has([data-testid="chatAvatarIcon-user"]) {
    background-color: var(--leia-surface) !important;
    border: 1px solid var(--leia-border) !important;
    border-radius: 2px !important;
}

/* ─── Chat messages — assistant ────────────────────── */

.stChatMessage:has([data-testid="chatAvatarIcon-assistant"]) {
    background-color: transparent !important;
    border-left: 2px solid var(--leia-gold) !important;
    border-radius: 0 !important;
    padding-left: 1rem;
}

/* ─── Chat message text ────────────────────────────── */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li {
    color: var(--leia-ivory) !important;
    font-size: 0.92rem !important;
    line-height: 1.75 !important;
}

/* ─── Bouton clear ─────────────────────────────────── */
.stButton > button {
    background-color:  var(--leia-surface) !important;
    color: var(--leia-muted) !important;
    border: 1px solid var(--leia-ivory) !important;
    font-family: 'Jost', sans-serif !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-radius: 2px !important;
    padding: 0.3rem 1rem !important;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    border-color: var(--leia-gold) !important;
    color: var(--leia-gold) !important;
}

/* ─── Success / info banners ───────────────────────── */
[data-testid="stAlert"] {
    background-color: var(--leia-surface) !important;
    border: 1px solid var(--leia-border) !important;
    border-left: 3px solid var(--leia-gold) !important;
    color: var(--leia-ivory) !important;
    border-radius: 2px !important;
    font-family: 'Jost', sans-serif !important;
    font-size: 0.82rem !important;
}

/* ─── Expander (Sources) ───────────────────────────── */
[data-testid="stExpander"] {
    background-color: var(--leia-surface) !important;
    border: 1px solid var(--leia-border) !important;
    border-radius: 2px !important;
}

[data-testid="stExpander"] summary {
    color: var(--leia-muted) !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.05em;
}

[data-testid="stExpander"] summary:hover {
    color: var(--leia-gold) !important;
}

/* ─── Scrollbar ────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--leia-grey); }
::-webkit-scrollbar-thumb { background: var(--leia-border); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--leia-gold); }

/* ─── Spinner text ─────────────────────────────────── */
[data-testid="stSpinner"] p {
    color: var(--leia-gold) !important;
    font-style: italic;
    font-family: 'Cormorant Garamond', serif !important;
}
</style>
"""


def inject_leia_style():
    """
    Appelle cette fonction UNE FOIS au début de ton app, après st.set_page_config().
    
    Exemple d'utilisation dans app_rag.py :
    
        from leia_style import inject_leia_style
        
        st.set_page_config(
            page_title="LÉIA Assistant",
            page_icon="◈",
            layout="wide"
        )
        
        inject_leia_style()   # ← ajouter cette ligne
    """
    import streamlit as st
    st.markdown(LEIA_CSS, unsafe_allow_html=True)
