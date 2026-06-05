# streaming_monitoring.py  –  Industrial RM Monitoring Dashboard
# Version with molar ratio classes, level gauges, and flow indicators
# Modifié : intervalle [1.1,1.2] = zone cible, pas de graphe, alertes uniquement pour extrêmes

import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ModelFinal import predire_classe_rm

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rapport Molaire Monitor",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS (identique sauf suppression de l'animation et ajustements) ────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;400;600;700&display=swap');

/* ── Root palette ── */
:root {
    --bg-main:    #0d1117;
    --bg-card:    #161b22;
    --bg-card2:   #1c2330;
    --border:     #30363d;
    --accent:     #58a6ff;
    --green:      #3fb950;
    --orange:     #d29922;
    --red:        #f85149;
    --text-pri:   #e6edf3;
    --text-sec:   #8b949e;
    --mono:       'Share Tech Mono', monospace;
    --sans:       'Barlow', sans-serif;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-main) !important;
    color: var(--text-pri) !important;
    font-family: var(--sans) !important;
}
[data-testid="stSidebar"] {
    background-color: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-pri) !important; }

/* Header */
.dash-header {
    background: linear-gradient(135deg, #0d1117 0%, #1c2330 100%);
    border-bottom: 1px solid var(--border);
    padding: 18px 28px 14px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 14px;
}
.dash-header .logo { font-size: 2.2rem; line-height: 1; }
.dash-header .titles { display: flex; flex-direction: column; }
.dash-header .main-title {
    font-family: var(--sans);
    font-weight: 700;
    font-size: 1.5rem;
    color: var(--text-pri);
    letter-spacing: .04em;
    margin: 0;
}
.dash-header .sub-title {
    font-family: var(--mono);
    font-size: .78rem;
    color: var(--accent);
    margin: 0;
    letter-spacing: .1em;
}
.dash-header .live-badge {
    margin-left: auto;
    display: flex;
    align-items: center;
    gap: 7px;
    font-family: var(--mono);
    font-size: .78rem;
    color: var(--green);
    border: 1px solid var(--green);
    padding: 4px 12px;
    border-radius: 20px;
}
.live-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--green);
    animation: pulse 1.4s ease-in-out infinite;
}
@keyframes pulse { 0%,100% { opacity:1; transform:scale(1); } 50% { opacity:.4; transform:scale(.7); } }

/* KPI cards */
.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px 22px;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content:'';
    position: absolute;
    top:0; left:0; right:0;
    height: 3px;
}
.kpi-card.normal::before  { background: var(--green); }
.kpi-card.warning::before { background: var(--orange); }
.kpi-card.critical::before{ background: var(--red); }
.kpi-card.neutral::before { background: var(--accent); }

.kpi-label {
    font-family: var(--mono);
    font-size: .72rem;
    color: var(--text-sec);
    letter-spacing: .1em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.kpi-value {
    font-family: var(--mono);
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--text-pri);
    line-height: 1;
}
.kpi-sub {
    font-family: var(--sans);
    font-size: .78rem;
    color: var(--text-sec);
    margin-top: 4px;
}

/* Sections */
.section-label {
    font-family: var(--mono);
    font-size: .72rem;
    color: var(--accent);
    letter-spacing: .14em;
    text-transform: uppercase;
    margin-bottom: 10px;
    border-left: 3px solid var(--accent);
    padding-left: 8px;
}

/* Variable rows */
.var-row {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 16px;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.var-name {
    font-family: var(--mono);
    font-size: .82rem;
    color: var(--text-sec);
}
.var-val {
    font-family: var(--mono);
    font-size: .95rem;
    color: var(--text-pri);
    font-weight: 600;
}
.var-label {
    font-family: var(--sans);
    font-size: .7rem;
    color: #6e7681;
    margin-top: 2px;
}

/* Alerts */
.alert-item {
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-family: var(--sans);
    font-size: .85rem;
    display: flex;
    align-items: flex-start;
    gap: 8px;
}
.alert-ok       { background: #0d2818; border: 1px solid var(--green); color: #56d364; }
.alert-warning  { background: #2b1f00; border: 1px solid var(--orange); color: #e3b341; }
.alert-critical { background: #2d0f0e; border: 1px solid var(--red); color: #ff7b72; }
.alert-info     { background: #1c2c3e; border: 1px solid var(--accent); color: #8bb9fe; }

/* Progress bar */
.prog-track {
    background: var(--border);
    border-radius: 4px;
    height: 6px;
    width: 100%;
    margin-top: 6px;
}
.prog-fill {
    height: 6px;
    border-radius: 4px;
    background: linear-gradient(90deg, var(--green), var(--accent));
    transition: width .4s ease;
}

/* Classe indicator (icône simple) */
.class-indicator {
    background: var(--bg-card);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    border: 1px solid var(--border);
}
.class-icon {
    font-size: 4rem;
    line-height: 1;
}
.class-name {
    font-family: var(--mono);
    font-size: 1.2rem;
    font-weight: 600;
    margin-top: 10px;
}
.class-desc {
    font-size: .8rem;
    color: var(--text-sec);
    margin-top: 8px;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* Sidebar buttons */
.stButton > button {
    width: 100%;
    background: var(--bg-card2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-pri) !important;
    font-family: var(--mono) !important;
    font-size: .82rem !important;
    border-radius: 6px !important;
    padding: 8px 0 !important;
    transition: border-color .2s, background .2s;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    background: #1c2c3e !important;
}
[data-testid="stSlider"] label {
    font-family: var(--mono) !important;
    font-size: .75rem !important;
    color: var(--text-sec) !important;
}
</style>
""", unsafe_allow_html=True)

# ── 1. LABELS ────────────────────────────────────────────────────────────────
LABELS = {
    "PI2022": "(mbar) pression de l'acide phosphorique dans le laveur acide",
    "TI2129": "(°C) temperature du laveur acide",
    "PI2023": "(barg) ecart a la pression atmospherique dans le laveur",
    "VI2921": "A remplir",
    "FIT2406": "(kg/h) debit massique de l'ammoniac dans l'evaporateur",
    "PIT2024": "(bar) pression dans l'evaporateur",
    "TIT2109": "(°C) temperature de l'ammoniac dans l'evaporateur",
    "DI2403": "(kg/m²) masse par unite de surface de H3PO4 entrant dans le reacteur AR201",
    "FIC2401": "(kg/h) debit massique de l'ammoniac entrant dans AR201",
    "LIC2218": "(m) niveau dans le reacteur AR201",
    "PI2032": "(bara) pression dans le reacteur AR201",
    "TI2101": "(°C) temperature dans le reacteur AR201",
    "FIC2402": "(kg/h) debit massique de l'ammoniac entrant dans AR202",
    "LI2215": "(m) niveau dans le reacteur AR202",
    "TI2116": "(°C) temperature dans le reacteur AR202",
    "FI5431": "(m³/h) debit volumique des eaux meres entrant dans AR202",
    "LIC2202": "Defluorination AR203",
    "LI2205": "Tampon AR204",
    "LI2207": "Tampon AR207",
    "LI2206": "Repulpage AR205",
    "LI2231": "Tampon AR225",
    "FQI2407": "Tampon AR225",
    "LIC5202": "Eaux meres EM AR501 - niveau",
    "TI5101": "Eaux meres EM AR501 - temperature",
    "FIC5430": "Eaux meres EM AR501 - debit"
}

# ── 2. DATA LOADING ──────────────────────────────────────────────────────────
@st.cache_data
def charger_db_reference():
    df = pd.read_csv('db_net_7.csv')
    cible = next((c for c in df.columns if c.lower() == 'rm'), None)
    if cible is None:
        st.error("Pas de colonne 'rm' dans db_net_7.csv")
        st.stop()
    if cible != 'rm':
        df.rename(columns={cible: 'rm'}, inplace=True)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = df[num_cols].reset_index(drop=True)
    return df

@st.cache_data
def charger_flux_monitoring():
    df = pd.read_csv('bd_monitoring_impute_median.csv', sep=';', decimal=',')
    cols_date = [c for c in df.columns if 'date' in c.lower() or 'heure' in c.lower()]
    if cols_date:
        df = df.drop(columns=cols_date)
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

@st.cache_data
def stats_reference(df):
    bins   = [-np.inf, 1.1, 1.16, 1.2, np.inf]
    labels = ['rm<1.1', '1.1≤rm<1.16', '1.16≤rm<1.2', 'rm≥1.2']
    df_t   = df.copy()
    df_t['classe'] = pd.cut(df_t['rm'], bins=bins, labels=labels, right=False)
    return {
        col: df_t.groupby('classe')[col].mean().to_dict()
        for col in df_t.columns
        if col not in ['rm', 'classe']
    }

@st.cache_data
def get_bounds_reference(df):
    bounds = {}
    for col in df.columns:
        if col == 'rm':
            continue
        bounds[col] = {
            'min': df[col].min(),
            'max': df[col].max(),
            'median': df[col].median()
        }
    return bounds

# ── 3. ACTION RULES (uniquement pour les classes extrêmes) ────────────────────
# On supprime toutes les règles pour les classes internes (1.1-1.2)
REGLES = {
    'TIT2109': {'rm<1.1':       ('plus_faible',  "Augmenter TIT2109")},
    'FI5431':  {'rm<1.1':       ('plus_faible',  "Augmenter FI5431")},
    'LI2231':  {'rm<1.1':       ('plus_élevée',  "Diminuer LI2231")},
    'DI2403':  {'rm≥1.2':       ('plus_faible',  "Diminuer débit d'acide")},
    'FIC2401': {'rm≥1.2':       ('plus_faible',  "Augmenter FIC2401")},
    'LIC2202': {'rm≥1.2':       ('plus_faible',  "Augmenter LIC2202")},
    'LI2205':  {'rm≥1.2':       ('plus_faible',  "Augmenter LI2205")},
    'LI2207':  {'rm≥1.2':       ('plus_faible',  "Augmenter LI2207")},
    'Densité1':{'rm≥1.2':       ('plus_élevée',  "Diminuer Densité1")},
    'pH1':     {'rm≥1.2':       ('plus_faible',  "Augmenter pH1")},
    'Densité2':{'rm≥1.2':       ('plus_faible',  "Augmenter Densité2")},
    'pH2':     {'rm≥1.2':       ('plus_faible',  "Augmenter pH2")},
    '%TS':     {'rm≥1.2':       ('plus_faible',  "Augmenter %TS")},
    'Densité': {'rm≥1.2':       ('plus_élevée',  "Diminuer Densité")},
    '%TS2':    {'rm≥1.2':       ('plus_faible',  "Augmenter %TS2")},
}

CLASS_MIDPOINT = {
    'rm<1.1':       1.05,
    '1.1≤rm<1.16':  1.13,
    '1.16≤rm<1.2':  1.18,
    'rm≥1.2':       1.25,
}

CLASS_LABEL_MAP = {
    0: '1.1≤rm<1.16',
    1: '1.16≤rm<1.2',
    2: 'rm<1.1',
    3: 'rm≥1.2',
}

def normaliser_label(pred_classe, pred_label):
    if isinstance(pred_classe, int):
        return CLASS_LABEL_MAP.get(pred_classe, pred_label)
    return pred_label

def generer_actions(pred_label, valeurs, stats):
    """Génère des actions uniquement pour les classes extrêmes (<1.1 ou >1.2)"""
    if pred_label in ('1.1≤rm<1.16', '1.16≤rm<1.2'):
        return [('ok', "Rapport molaire dans la zone cible – aucune action requise")]
    actions = []
    for var, regle in REGLES.items():
        if var not in valeurs or pd.isna(valeurs[var]):
            continue
        # pred_label est soit 'rm<1.1' soit 'rm≥1.2'
        if pred_label in regle:
            tendance, action = regle[pred_label]
            # Pour une action corrective, on compare avec la moyenne de la zone cible (1.1-1.16)
            moy_cible = stats.get(var, {}).get('1.1≤rm<1.16')
            if moy_cible is not None:
                val_act = valeurs[var]
                if (tendance == 'plus_faible'  and val_act < moy_cible) or \
                   (tendance == 'plus_élevée'  and val_act > moy_cible):
                    severity = 'critical'  # car extrême
                    actions.append((severity, f"{var} = {val_act:.3f}  (cible {moy_cible:.3f}) → {action}"))
    return actions if actions else [('warning', "Aucune action spécifique identifiée")]

def severity_class(pred_label):
    # Désormais, toute la zone [1.1, 1.2] est 'normal'
    if pred_label in ('1.1≤rm<1.16', '1.16≤rm<1.2'):
        return 'normal'
    else:
        return 'critical'  # extrêmes

def render_current_row(row, idx):
    st.markdown('<div class="section-label">SUIVI LIGNE PAR LIGNE</div>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-family:\'Share Tech Mono\',monospace;font-size:.78rem;color:#8b949e;margin:0 0 10px">Ligne {idx + 1}</p>', unsafe_allow_html=True)
    st.dataframe(row.to_frame(name='Valeur').reset_index().rename(columns={'index': 'Variable'}), height=260)

# ── 4. OUTLIER HANDLING ──────────────────────────────────────────────────────
def traiter_outliers(X_original, bounds):
    X = X_original.copy()
    messages = []
    temperatures = ['TI2129', 'TIT2109', 'TI2101', 'TI2116']
    niveaux = ['LIC2218', 'LI2215', 'LIC2202']
    debits = ['FIT2406', 'FIC2401', 'FIC2402', 'FI5431', 'DI2403']
    for var, val in X.items():
        if var not in bounds or pd.isna(val):
            continue
        b = bounds[var]
        mini, maxi, mediane = b['min'], b['max'], b['median']
        if val < mini or val > maxi:
            if var in temperatures:
                X[var] = mediane
                messages.append(("info", f"⚠ {var} : {val:.3f} hors bornes → remplacé par médiane ({mediane:.3f})"))
            elif var in niveaux or var in debits:
                continue
            else:
                new_val = mini if val < mini else maxi
                X[var] = new_val
                messages.append(("info", f"⚠ {var} : {val:.3f} → ramené à {new_val:.3f} (borne)"))
    return X, messages

# ── 5. INDICATEUR DE CLASSE (icône unique, pas de graphe) ─────────────────────
def render_class_indicator(pred_label):
    icon = "🎯" if pred_label in ('1.1≤rm<1.16', '1.16≤rm<1.2') else "⚠️" if pred_label == 'rm<1.1' else "🔴"
    color = "#3fb950" if pred_label in ('1.1≤rm<1.16', '1.16≤rm<1.2') else "#f85149"
    st.markdown(f"""
    <div class="class-indicator" style="border-color:{color}">
        <div class="class-icon">{icon}</div>
        <div class="class-name" style="color:{color}">{pred_label}</div>
        <div class="class-desc">Rapport molaire prédit</div>
    </div>
    """, unsafe_allow_html=True)

# ── 6. INDICATEURS DE PROCESSUS (niveaux, débits) ────────────────────────────
def render_process_indicators(X, bounds):
    st.markdown('<div class="section-label">ÉTAT PROCÉDÉ</div>', unsafe_allow_html=True)
    if 'LIC2218' in X and 'LIC2218' in bounds:
        val = X['LIC2218']
        b = bounds['LIC2218']
        pct = (val - b['min']) / (b['max'] - b['min']) * 100
        pct = max(0, min(100, pct))
        st.markdown(f'<div class="var-row"><span class="var-name">AR201 - Niveau</span><span class="var-val">{pct:.1f}%</span></div>', unsafe_allow_html=True)
        st.progress(pct/100, text=f"{val:.2f} {LABELS.get('LIC2218', 'm')}")
    if 'LI2215' in X and 'LI2215' in bounds:
        val = X['LI2215']
        b = bounds['LI2215']
        pct = (val - b['min']) / (b['max'] - b['min']) * 100
        pct = max(0, min(100, pct))
        st.markdown(f'<div class="var-row"><span class="var-name">AR202 - Niveau</span><span class="var-val">{pct:.1f}%</span></div>', unsafe_allow_html=True)
        st.progress(pct/100, text=f"{val:.2f} {LABELS.get('LI2215', 'm')}")
    if 'DI2403' in X and 'DI2403' in bounds:
        val = X['DI2403']
        b = bounds['DI2403']
        pct = (val - b['min']) / (b['max'] - b['min']) * 100
        pct = max(0, min(100, pct))
        st.markdown(f'<div class="var-row"><span class="var-name">Débit acide H₃PO₄</span><span class="var-val">{val:.2f} {LABELS.get("DI2403", "kg/m²")}</span></div>', unsafe_allow_html=True)
        st.progress(pct/100, text=f"Référence: min {b['min']:.2f} / max {b['max']:.2f}")
    if 'FIC2401' in X and 'FIC2401' in bounds:
        val = X['FIC2401']
        b = bounds['FIC2401']
        pct = (val - b['min']) / (b['max'] - b['min']) * 100
        pct = max(0, min(100, pct))
        st.markdown(f'<div class="var-row"><span class="var-name">Débit ammoniac (NH₃)</span><span class="var-val">{val:.2f} {LABELS.get("FIC2401", "kg/h")}</span></div>', unsafe_allow_html=True)
        st.progress(pct/100, text=f"Référence: min {b['min']:.2f} / max {b['max']:.2f}")

# ── 7. KPI CARDS (sans graphe, adapté) ──────────────────────────────────────
def render_kpis(pred_label, n_alerts, is_manual=False):
    sev = severity_class(pred_label)
    status_text = "✅ DANS LA CIBLE" if sev == 'normal' else "🔴 HORS CIBLE"
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(
            f'<div class="kpi-card {sev}"><div class="kpi-label">Rapport molaire</div>'
            f'<div class="kpi-value" style="font-size:1.2rem">{pred_label}</div>'
            f'<div class="kpi-sub">Classe prédite</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(
            f'<div class="kpi-card {sev}"><div class="kpi-label">Statut</div>'
            f'<div class="kpi-value" style="font-size:1.3rem">{status_text}</div>'
            f'<div class="kpi-sub">Recommandation</div></div>', unsafe_allow_html=True)
    with k3:
        alert_color = 'critical' if n_alerts > 3 else ('warning' if n_alerts > 0 else 'normal')
        st.markdown(
            f'<div class="kpi-card {alert_color}"><div class="kpi-label">Alertes actives</div>'
            f'<div class="kpi-value">{n_alerts}</div><div class="kpi-sub">Ce cycle</div></div>',
            unsafe_allow_html=True)

# ── 8. ALERTS PANEL ──────────────────────────────────────────────────────────
def render_alerts(actions, outlier_messages=None):
    st.markdown('<div class="section-label">RECOMMANDATIONS</div>', unsafe_allow_html=True)
    icon_map = {'ok': '✅', 'warning': '⚠', 'critical': '🔴', 'info': 'ℹ️'}
    for sev, msg in actions:
        css = f'alert-{sev}'
        icon = icon_map.get(sev, 'ℹ')
        st.markdown(f'<div class="alert-item {css}"><span>{icon}</span><span>{msg}</span></div>', unsafe_allow_html=True)
    if outlier_messages:
        st.markdown('<div class="section-label" style="margin-top:16px">DÉTECTION OUTLIERS</div>', unsafe_allow_html=True)
        for sev, msg in outlier_messages:
            css = f'alert-{sev}'
            icon = icon_map.get(sev, 'ℹ')
            st.markdown(f'<div class="alert-item {css}"><span>{icon}</span><span>{msg}</span></div>', unsafe_allow_html=True)

# ── 9. SIDEBAR ───────────────────────────────────────────────────────────────
def render_sidebar(n_rows):
    st.sidebar.markdown(
        '<p style="font-family:\'Share Tech Mono\',monospace;font-size:.72rem;color:#58a6ff;'
        'letter-spacing:.12em;text-transform:uppercase;margin-bottom:16px">⚙ CONTROL PANEL</p>',
        unsafe_allow_html=True,
    )
    input_mode = st.sidebar.radio("Mode d'entrée", ["Streaming (CSV)", "Manuelle"], index=0, key='input_mode')
    if input_mode == "Streaming (CSV)":
        c1, c2, c3 = st.sidebar.columns(3)
        start  = c1.button("▶ Start")
        stop   = c2.button("⏸ Stop")
        reset  = c3.button("↺ Reset")
        st.sidebar.markdown('<hr style="border-color:#30363d;margin:16px 0">', unsafe_allow_html=True)
        freq    = st.sidebar.slider("Intervalle (s)", 1, 10, 5, key='freq')
        rm_min  = st.sidebar.slider("Seuil bas du rapport", 0.90, 1.10, 1.10, 0.01, key='rm_min')
        rm_max  = st.sidebar.slider("Seuil haut du rapport", 1.10, 1.40, 1.20, 0.01, key='rm_max')
        st.sidebar.markdown('<hr style="border-color:#30363d;margin:16px 0">', unsafe_allow_html=True)
        st.sidebar.markdown('<p style="font-family:\'Share Tech Mono\',monospace;font-size:.7rem;color:#8b949e;letter-spacing:.08em">VARIABLES AFFICHÉES</p>', unsafe_allow_html=True)
        all_vars = list(REGLES.keys())
        selected_vars = st.sidebar.multiselect(
        "Choisir", all_vars,
        default=['DI2403', 'pH2', 'FIC2401', 'Densité2'],
        key='sel_vars', label_visibility='collapsed'
        )
        if 'idx' in st.session_state and n_rows > 0:
           pct = int(st.session_state.get('idx', 0) / n_rows * 100)
           st.sidebar.markdown(f'<p style="font-family:\'Share Tech Mono\',monospace;font-size:.7rem;color:#8b949e;margin-top:16px">PROGRESSION: {pct}%</p><div class="prog-track"><div class="prog-fill" style="width:{pct}%"></div></div>', unsafe_allow_html=True)
        st.sidebar.markdown('<p style="font-family:\'Share Tech Mono\',monospace;font-size:.65rem;color:#30363d;margin-top:24px">RAPPORT MOLAIRE v2.3 · XGBoost+LightGBM</p>', unsafe_allow_html=True)
        return input_mode, start, stop, reset, freq, rm_min, rm_max, selected_vars
    else:
        return input_mode, None, None, None, None, None, None, None

# ── 10. FORMULAIRE MANUEL ────────────────────────────────────────────────────
def render_manual_form(bounds, stats):
    st.markdown('<div class="section-label">SAISIE DES VARIABLES</div>', unsafe_allow_html=True)
    with st.form("manual_input_form"):
        cols = st.columns(3)
        user_input = {}
        variables = sorted(bounds.keys())
        for i, var in enumerate(variables):
            col = cols[i % 3]
            default_val = bounds[var]['median']
            label_display = f"{var}\n{LABELS.get(var, '')}" if var in LABELS else var
            user_input[var] = col.number_input(label_display, value=float(default_val), format="%.4f", key=f"manual_{var}")
        submitted = st.form_submit_button("🔮 Prédire le rapport molaire", use_container_width=True)
    return user_input, submitted

# ── 11. MAIN ─────────────────────────────────────────────────────────────────
def main():
    with st.spinner("Initialisation..."):
        df_ref = charger_db_reference()
        df_flux = charger_flux_monitoring()
        stats = stats_reference(df_ref)
        bounds = get_bounds_reference(df_ref)
    try:
        sample_X = {col: bounds[col]['median'] for col in bounds.keys()}
        predire_classe_rm(sample_X)
    except Exception as e:
        st.error(f"Modèle introuvable : {e}. Lancez d'abord ModelFinal.py")
        st.stop()

    sidebar_result = render_sidebar(len(df_flux))
    input_mode = sidebar_result[0]
    if input_mode == "Streaming (CSV)":
        _, start, stop, reset, freq, rm_min, rm_max, sel_vars = sidebar_result
        # Session state
        if 'run' not in st.session_state: st.session_state.run = False
        if 'idx' not in st.session_state: st.session_state.idx = 0
        if 'last_label' not in st.session_state: st.session_state.last_label = '1.1≤rm<1.16'
        if 'last_actions' not in st.session_state: st.session_state.last_actions = [('ok','—')]
        if 'last_outliers' not in st.session_state: st.session_state.last_outliers = []
        if 'last_X' not in st.session_state: st.session_state.last_X = {}
        if 'last_row' not in st.session_state: st.session_state.last_row = pd.Series(dtype='float64')

        if start:
            st.session_state.run = True
            if st.session_state.idx >= len(df_flux):
                st.session_state.idx = 0
        if stop: st.session_state.run = False
        if reset:
            st.session_state.run = False
            st.session_state.idx = 0
            st.session_state.last_label = '1.1≤rm<1.16'
            st.session_state.last_actions = [('ok','—')]
            st.session_state.last_outliers = []
            st.session_state.last_X = {}
            st.session_state.last_row = pd.Series(dtype='float64')

        render_header(st.session_state.run, is_manual=False)

        if st.session_state.run and st.session_state.idx < len(df_flux):
            i = st.session_state.idx
            row = df_flux.iloc[i]
            X_raw = row.dropna().to_dict()
            X_corrige, outlier_msgs = traiter_outliers(X_raw, bounds)
            classe, raw_label, _ = predire_classe_rm(X_corrige)
            pred_label = normaliser_label(classe, raw_label)
            actions = generer_actions(pred_label, X_corrige, stats)

            st.session_state.last_label = pred_label
            st.session_state.last_actions = actions
            st.session_state.last_outliers = outlier_msgs
            st.session_state.last_X = X_corrige
            st.session_state.last_row = row
            st.session_state.idx += 1
        elif st.session_state.run and st.session_state.idx >= len(df_flux):
            st.session_state.run = False
            st.balloons()

        pred_label = st.session_state.last_label
        actions = st.session_state.last_actions
        outlier_msgs = st.session_state.last_outliers
        X = st.session_state.last_X
        current_row = st.session_state.get('last_row', pd.Series(dtype='float64'))
        n_alerts = sum(1 for s, _ in actions if s in ('warning', 'critical'))

        st.markdown('<div style="margin-bottom:20px">', unsafe_allow_html=True)
        render_kpis(pred_label, n_alerts, is_manual=False)
        st.markdown('</div><div style="height:4px"></div>', unsafe_allow_html=True)

        col_left, col_right = st.columns([1, 1], gap='medium')
        with col_left:
            render_class_indicator(pred_label)
            st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
            if not current_row.empty:
                render_current_row(current_row, st.session_state.idx - 1)
        with col_right:
            render_alerts(actions, outlier_msgs)
            st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
            render_process_indicators(X, bounds)
            st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
            render_variables(X, sel_vars)

        if not st.session_state.run and st.session_state.idx == 0:
            st.info("▶ Cliquez sur **Start** dans le panneau de contrôle.")
        elif not st.session_state.run and st.session_state.idx > 0:
            st.success(f"✔ Simulation terminée — {st.session_state.idx} cycles traités.")
        if st.session_state.run:
            time.sleep(freq)
            st.rerun()
    else:  # Mode manuel
        render_header(running=False, is_manual=True)
        user_input, submitted = render_manual_form(bounds, stats)
        if submitted:
            X_corrige, outlier_msgs = traiter_outliers(user_input, bounds)
            classe, raw_label, _ = predire_classe_rm(X_corrige)
            pred_label = normaliser_label(classe, raw_label)
            actions = generer_actions(pred_label, X_corrige, stats)
            n_alerts = sum(1 for s, _ in actions if s in ('warning', 'critical'))
            st.markdown('<div style="margin-bottom:20px">', unsafe_allow_html=True)
            render_kpis(pred_label, n_alerts, is_manual=True)
            st.markdown('</div><div style="height:4px"></div>', unsafe_allow_html=True)
            col_left, col_right = st.columns([1, 1], gap='medium')
            with col_left:
                render_class_indicator(pred_label)
                st.markdown('<div class="section-label">VALEURS SAISIES (après correction)</div>', unsafe_allow_html=True)
                for var, val in X_corrige.items():
                    label = LABELS.get(var, "")
                    st.markdown(f'<div class="var-row"><div><span class="var-name">{var}</span>{f"<div class=\"var-label\">{label}</div>" if label else ""}</div><span class="var-val">{val:.4f}</span></div>', unsafe_allow_html=True)
            with col_right:
                render_alerts(actions, outlier_msgs)
        else:
            st.info("✏️ Remplissez le formulaire et cliquez sur **Prédire**.")

def render_variables(X, selected_vars):
    st.markdown('<div class="section-label">VARIABLES CLÉS</div>', unsafe_allow_html=True)
    shown = [v for v in selected_vars if v in X and not pd.isna(X.get(v))]
    if not shown:
        st.markdown('<p style="color:#8b949e;font-size:.8rem">Aucune variable sélectionnée.</p>', unsafe_allow_html=True)
        return
    for var in shown:
        val = X[var]
        label = LABELS.get(var, "")
        st.markdown(f'<div class="var-row"><div><span class="var-name">{var}</span>{f"<div class=\"var-label\">{label}</div>" if label else ""}</div><span class="var-val">{val:.4f}</span></div>', unsafe_allow_html=True)

def render_header(running, is_manual):
    if is_manual:
        badge = '<span class="live-badge" style="color:#58a6ff;border-color:#58a6ff">✍️ MANUAL</span>'
    else:
        badge = ('<span class="live-badge"><span class="live-dot"></span>LIVE</span>' if running else
                 '<span class="live-badge" style="color:#8b949e;border-color:#30363d"><span class="live-dot" style="background:#8b949e;animation:none"></span>IDLE</span>')
    st.markdown(
        f'<div class="dash-header">'
        f'<span class="logo">⚗️</span>'
        f'<div class="titles">'
        f'<p class="main-title">RAPPORT MOLAIRE MONITOR</p>'
        f'<p class="sub-title">PRODUCTION ACIDE PHOSPHORIQUE · CONTRÔLE PRÉDICTIF</p>'
        f'</div>'
        f'{badge}'
        f'</div>',
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()