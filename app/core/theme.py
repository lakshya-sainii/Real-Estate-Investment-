"""Estate IQ design system: obsidian, mint and chrome; Talina + Manrope."""
import streamlit as st
import plotly.graph_objects as go
from .fonts import TALINA_B64

BG = "#050707"
PAPER = "#0B0F0E"
INK = "#F1F5F3"
MINT = "#55F5B8"
MINT_DEEP = "#20C990"
CHROME = "#77847E"
GRAPHITE = "#A1ADA7"
CORAL = "#FF7D7D"
COLORWAY = [MINT, INK, MINT_DEEP, CHROME, "#B8C4BF", CORAL]
MINT_SCALE = ["#0B100F", "#102A22", "#124837", "#176E50", "#20B884"]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;600;700;800&display=swap');
@font-face{font-family:'Talina';src:url(data:font/otf;base64,__TALINA__) format('opentype');font-display:swap;}
:root{--bg:#050707;--panel:#0B0F0E;--panel2:#101614;--text:#F1F5F3;--mint:#55F5B8;--mint-deep:#20C990;--chrome:#77847E;--graphite:#A1ADA7;--hair:rgba(197,220,209,.14);--black:#050707;}
html,body,[class*="css"],.stApp{font-family:'Manrope',sans-serif;color:var(--text);}
.stApp{background:radial-gradient(900px 560px at 100% -5%,rgba(32,201,144,.09),transparent 60%),radial-gradient(760px 480px at -8% 42%,rgba(85,245,184,.035),transparent 68%),var(--bg);}
[data-testid="stHeader"],[data-testid="stToolbar"]{background:transparent;}
[data-testid="stAppViewContainer"]{background:transparent;}
.block-container{padding-top:1.1rem;padding-bottom:3rem;max-width:1320px;}
h1,h2,h3{font-family:'Talina','Manrope',sans-serif !important;font-weight:400 !important;letter-spacing:-.012em;color:var(--text);}
h1{font-size:3.2rem !important;line-height:1 !important;}
h4,p,li,label,.stMarkdown{color:var(--text);}
.lbl{font-size:.66rem;letter-spacing:.22em;text-transform:uppercase;font-weight:700;color:var(--graphite);}
.mut{color:var(--graphite);}
mark,.hl{background:linear-gradient(transparent 62%,rgba(85,245,184,.72) 62%);color:inherit;padding:0 .08em;}

/* navigation and the single Estate IQ brand mark */
.st-key-topnav{border-bottom:1px solid var(--hair);padding:.2rem 0 1rem;margin-bottom:1.8rem;}
.brand{font-family:'Talina','Manrope',sans-serif;font-size:1.55rem;line-height:1;display:flex;align-items:center;gap:.72rem;width:fit-content;}
.brand-mark{position:relative;display:inline-block;width:29px;height:29px;transform:rotate(45deg);border:1px solid rgba(85,245,184,.65);border-radius:5px;transition:transform .45s cubic-bezier(.2,.8,.2,1),box-shadow .45s ease;box-shadow:0 0 18px rgba(85,245,184,.09);}
.brand-mark:before,.brand-mark:after{content:"";position:absolute;border:1px solid var(--mint);border-radius:2px;}
.brand-mark:before{inset:6px;border-right-color:transparent;border-bottom-color:transparent;}
.brand-mark:after{width:5px;height:5px;right:5px;top:5px;background:var(--mint);border:0;box-shadow:0 0 12px rgba(85,245,184,.7);}
.brand:hover .brand-mark{transform:rotate(135deg) scale(1.06);box-shadow:0 0 24px rgba(85,245,184,.25);}
.brand-text{transform:translateY(1px);}
.st-key-topnav [data-testid="stPageLink-NavLink"]{background:transparent !important;padding:.35rem .2rem;justify-content:center;border-radius:0;}
.st-key-topnav [data-testid="stPageLink-NavLink"] p{font-size:.62rem !important;letter-spacing:.17em;text-transform:uppercase;font-weight:700;color:var(--graphite);white-space:nowrap;}
.st-key-topnav [data-testid="stPageLink-NavLink"]:hover p{color:var(--mint);}
.st-key-topnav [data-testid="stPageLink-NavLink"][aria-current="page"] p{color:var(--text);box-shadow:inset 0 -2px 0 var(--mint-deep);padding-bottom:.25rem;}

/* landing hero: original composition and typography retained */
.st-key-hero{padding:.4rem 0 0;}
.hero-h{font-family:'Talina','Manrope',sans-serif;font-size:4.15rem;line-height:.98;letter-spacing:-.02em;margin:.5rem 0 1.4rem;}
.hero-h .hl{text-shadow:0 0 24px rgba(85,245,184,.14);}
.hero-sub{font-size:1.15rem;font-weight:700;line-height:1.45;margin-bottom:.8rem;}
.hero-p{font-size:.92rem;color:var(--graphite);line-height:1.6;max-width:440px;}
.idx{display:grid;grid-template-columns:1fr 1fr;gap:.5rem 2rem;margin:1.3rem 0 1.4rem;max-width:420px;}
.idx div{font-size:.66rem;letter-spacing:.2em;text-transform:uppercase;font-weight:700;color:var(--graphite);}
.idx b{color:var(--chrome);font-weight:600;margin-right:.55rem;}
.rule{width:56px;height:1px;background:var(--mint-deep);margin:1.4rem 0 1.2rem;}
.st-key-cta [data-testid="stPageLink-NavLink"]{background:var(--mint) !important;border:1px solid var(--mint);border-radius:3px;padding:1rem 1.5rem;width:fit-content;transition:transform .2s ease,box-shadow .2s ease;}
.st-key-cta [data-testid="stPageLink-NavLink"] p{color:var(--black) !important;font-size:.7rem !important;letter-spacing:.22em;text-transform:uppercase;font-weight:800;}
.st-key-cta [data-testid="stPageLink-NavLink"]:hover{transform:translateY(-2px);box-shadow:0 10px 30px rgba(85,245,184,.14);}
.hero-art{position:relative;border:1px solid var(--hair);border-radius:8px;overflow:hidden;background:#000;box-shadow:0 28px 70px rgba(0,0,0,.36);}
.hero-art img{display:block;width:100%;height:540px;object-fit:cover;object-position:center 48%;transition:transform 1.1s cubic-bezier(.2,.7,.2,1),filter .5s ease;filter:saturate(.9) contrast(1.04);}
.hero-art:hover img{transform:scale(1.025);filter:saturate(1.05) contrast(1.05);}
.hero-art:after{content:"";position:absolute;inset:0;pointer-events:none;background:linear-gradient(180deg,transparent 70%,rgba(5,7,7,.72));}
.art-caption{position:absolute;z-index:2;left:18px;right:18px;bottom:15px;display:flex;justify-content:space-between;color:#C4D0CA;font-size:.58rem;letter-spacing:.21em;text-transform:uppercase;font-weight:700;}
.art-caption span:last-child{color:var(--mint);}

/* stat strip */
.stats{display:grid;grid-template-columns:1.25fr repeat(4,1fr) 1.25fr;border-top:1px solid var(--hair);margin-top:2.2rem;padding-top:1.1rem;align-items:center;}
.stats .cell{border-left:1px solid var(--hair);padding:.1rem 1.1rem;}
.stats .cell:first-child{border-left:none;padding-left:0;}
.stats .n{font-size:2.15rem;font-weight:300;letter-spacing:-.02em;line-height:1.05;}
.stats .t{font-size:.62rem;letter-spacing:.2em;text-transform:uppercase;font-weight:700;color:var(--graphite);margin-top:.2rem;}
.stats .t em{font-style:normal;color:var(--mint);}
.stats .big{font-size:1.15rem;font-weight:600;line-height:1.3;}
@media (max-width:900px){.stats{grid-template-columns:1fr 1fr;gap:1rem}.hero-h{font-size:3.3rem}.hero-art img{height:440px}}

/* cards and sections */
.card{min-height:360px;background:linear-gradient(145deg,rgba(18,25,22,.9),rgba(8,12,11,.94));border:1px solid var(--hair);border-radius:7px;padding:1.3rem 1.4rem;height:100%;position:relative;transition:transform .22s ease,border-color .22s ease,box-shadow .22s ease;}
.card:before{content:"";position:absolute;left:0;top:0;right:0;height:2px;border-radius:7px 7px 0 0;background:linear-gradient(90deg,transparent,#52665D 35%,#D5E1DB 55%,#34473F 80%,var(--mint));}
.card:hover{transform:translateY(-3px);border-color:rgba(85,245,184,.28);box-shadow:0 18px 44px rgba(0,0,0,.28);}
.card h4{font-family:'Talina','Manrope',sans-serif;font-weight:400;font-size:1.45rem;margin:.5rem 0 .4rem;line-height:1.05;}
.card p,.card li{font-size:.88rem;color:var(--graphite);line-height:1.55;}
.spec{display:flex;justify-content:space-between;border-top:1px solid var(--hair);padding:.5rem 0;font-size:.82rem;gap:1rem;}
.spec span:first-child{color:var(--graphite);letter-spacing:.12em;text-transform:uppercase;font-size:.64rem;font-weight:700;padding-top:.15rem;}
.sec{margin:2.8rem 0 1.1rem;}
.sec .lbl{margin-bottom:.35rem;}
.sec h2{font-size:2.5rem !important;line-height:1 !important;margin:0 !important;}
.qcard{margin-bottom:.6rem;border:1px solid var(--hair);border-radius:7px;padding:1rem 1.1rem;background:rgba(15,22,19,.82);min-height:118px;}
.qcard .p{font-size:.62rem;letter-spacing:.2em;text-transform:uppercase;font-weight:800;color:var(--mint);}
.qcard .q{font-weight:600;font-size:.98rem;margin-top:.35rem;line-height:1.35;}
.insight{border:1px solid var(--hair);border-left:3px solid var(--mint-deep);background:rgba(14,21,18,.84);border-radius:5px;padding:.85rem 1rem;margin:.2rem 0 1.3rem;}
.insight .tag{font-size:.6rem;letter-spacing:.22em;text-transform:uppercase;font-weight:800;color:var(--mint);margin-bottom:.25rem;}
.insight div:last-child{font-size:.88rem;line-height:1.5;color:#D7DFDA;}

/* verdict */
.verdict{border-radius:7px;padding:1.5rem 1.6rem;border:1px solid var(--hair);position:relative;overflow:hidden;background:var(--panel);}
.verdict.good{background:linear-gradient(135deg,rgba(85,245,184,.94),rgba(32,201,144,.68));border-color:rgba(85,245,184,.8);color:var(--black);}
.verdict.bad{background:linear-gradient(135deg,rgba(45,25,26,.95),rgba(21,14,15,.96));border-color:rgba(255,125,125,.4);}
.verdict .big{font-family:'Talina','Manrope',sans-serif;font-size:2.7rem;line-height:1;margin:.5rem 0 .8rem;}
.verdict.good .big{color:var(--black);text-shadow:0 0 18px rgba(255,255,255,.2);}
.pill{display:inline-block;padding:.22rem .7rem;border:1px solid currentColor;border-radius:999px;font-size:.62rem;letter-spacing:.16em;text-transform:uppercase;font-weight:800;margin-right:.4rem;background:rgba(5,7,7,.09);}
.crit{display:flex;justify-content:space-between;align-items:center;gap:1rem;padding:.7rem 0;border-bottom:1px solid var(--hair);}
.crit:last-child{border-bottom:none;}
.crit b{font-size:.9rem;} .crit small{color:var(--graphite);font-size:.78rem;}
.ok,.no{font-size:.6rem;letter-spacing:.18em;text-transform:uppercase;font-weight:800;padding:.25rem .6rem;border-radius:2px;white-space:nowrap;}
.ok{background:var(--mint);color:var(--black);} .no{background:#261719;color:#FFAAAA;border:1px solid rgba(255,125,125,.25);}
.property-visual{position:relative;min-height:430px;height:430px;border-radius:8px;overflow:hidden;background:#000 center 48%/cover no-repeat;border:1px solid var(--hair);}
.property-visual:after{content:"";position:absolute;inset:0;background:linear-gradient(180deg,rgba(0,0,0,.08),rgba(5,7,7,.82));}
.property-visual .pv-badge{position:absolute;z-index:2;top:16px;left:16px;padding:.55rem .85rem;border-radius:3px;background:var(--mint);color:var(--black);font-size:.62rem;letter-spacing:.18em;text-transform:uppercase;font-weight:800;}
.property-visual .pv-caption{position:absolute;z-index:2;left:18px;bottom:18px;font-family:'Talina','Manrope',sans-serif;font-size:1.7rem;color:white;}

/* controls */
.stButton>button,[data-testid="stFormSubmitButton"] button{background:var(--mint);color:var(--black);border:1px solid var(--mint);border-radius:3px;font-weight:800;letter-spacing:.16em;text-transform:uppercase;font-size:.7rem;padding:.85rem 1.3rem;transition:transform .2s ease,box-shadow .2s ease;}
.stButton>button:hover,[data-testid="stFormSubmitButton"] button:hover{background:#83F9CA;color:var(--black);border-color:#83F9CA;transform:translateY(-1px);box-shadow:0 8px 24px rgba(85,245,184,.12);}
.stButton>button p,[data-testid="stFormSubmitButton"] button p{font-size:.7rem !important;color:var(--black);}
div[data-baseweb="tag"],span[data-baseweb="tag"],[data-testid="stMultiSelectTag"]{background-color:#17372C !important;color:var(--mint) !important;}
[data-baseweb="tag"] span,[data-testid="stMultiSelectTag"] span{color:var(--mint) !important;}
[data-testid="stMetric"]{background:rgba(14,21,18,.82);border:1px solid var(--hair);border-radius:7px;padding:.9rem 1.05rem;}
[data-testid="stMetricLabel"] p{font-size:.62rem !important;letter-spacing:.18em;text-transform:uppercase;font-weight:800;color:var(--graphite);}
[data-testid="stMetricValue"]{font-weight:300;letter-spacing:-.02em;color:var(--text);}
div[data-testid="stForm"]{border:1px solid var(--hair);border-radius:7px;background:rgba(12,18,16,.8);padding:1.3rem;}
.stTabs [data-baseweb="tab"]{font-size:.64rem;letter-spacing:.14em;text-transform:uppercase;font-weight:800;color:var(--graphite);}
.stTabs [data-baseweb="tab"][aria-selected="true"]{color:var(--text);}
.stTabs [data-baseweb="tab-highlight"]{background:var(--mint);height:2px;}
.stTabs [data-baseweb="tab-list"]{border-bottom:1px solid var(--hair);gap:.4rem;}
[data-testid="stExpander"]{border:1px solid var(--hair);border-radius:7px;background:rgba(12,18,16,.82);}
[data-testid="stChatMessage"]{background:rgba(14,21,18,.84);border:1px solid var(--hair);border-radius:7px;}
[data-testid="stDataFrame"]{border:1px solid var(--hair);border-radius:6px;overflow:hidden;}
[data-baseweb="input"],[data-baseweb="select"]>div,[data-baseweb="textarea"]{background-color:#0D1311;border-color:var(--hair);}
[data-baseweb="popover"]>div,[data-baseweb="menu"]{background:#0D1311 !important;color:var(--text) !important;}
[data-baseweb="slider"]>div>div>div{background:var(--mint-deep) !important;}
.stCaption,[data-testid="stCaptionContainer"]{color:var(--graphite) !important;}
.note{font-size:.78rem;color:var(--graphite);border-top:1px solid var(--hair);padding-top:.6rem;margin-top:.8rem;}
.footer{margin-top:3rem;padding-top:1rem;border-top:1px solid var(--hair);color:var(--graphite);font-size:.68rem;letter-spacing:.13em;text-transform:uppercase;display:flex;justify-content:space-between;flex-wrap:wrap;gap:.6rem;}
.modebar-container{display:none !important;}
[class*="st-key-bq_"] button,[class*="st-key-fu_"] button{background:#101714;color:var(--text);border:1px solid var(--hair);text-transform:none;letter-spacing:0;font-weight:600;text-align:left;justify-content:flex-start;height:auto;min-height:3rem;padding:.7rem .9rem;border-radius:4px;box-shadow:none;}
[class*="st-key-bq_"] button p,[class*="st-key-fu_"] button p{text-transform:none !important;letter-spacing:0 !important;font-size:.84rem !important;font-weight:600;white-space:normal;text-align:left;line-height:1.35;}
[class*="st-key-bq_"] button:hover,[class*="st-key-fu_"] button:hover{background:#17372C;color:var(--mint);border-color:var(--mint-deep);}
.atlas-note{border:1px solid var(--hair);border-radius:7px;padding:1rem 1.2rem;background:radial-gradient(circle at 90% 10%,rgba(85,245,184,.08),transparent 45%),#0C1210;}
.fade{animation:fadeUp .8s ease both;}@keyframes fadeUp{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:none}}
</style>
"""


def inject_css():
    st.markdown(CSS.replace("__TALINA__", TALINA_B64), unsafe_allow_html=True)


def section(num: str, title: str, sub: str = ""):
    st.markdown(f'<div class="sec"><div class="lbl">{num}</div><h2>{title}</h2>' + (f'<div class="mut" style="margin-top:.5rem;font-size:.92rem">{sub}</div>' if sub else "") + "</div>", unsafe_allow_html=True)


def page_title(label: str, title: str, sub: str = ""):
    st.markdown(f'<div class="lbl">{label}</div><h1 style="margin:.3rem 0 .6rem">{title}</h1>' + (f'<div class="mut" style="max-width:680px">{sub}</div>' if sub else ""), unsafe_allow_html=True)
    st.write("")


def insight(title: str, text: str):
    st.markdown(f'<div class="insight"><div class="tag">Read on the data · {title}</div><div>{text}</div></div>', unsafe_allow_html=True)


def note(text="Illustrative model on a synthetic-style dataset with assumed growth rates. Not financial advice."):
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def footer():
    st.markdown('<div class="footer"><span>Estate IQ · Real Estate Investment Advisor</span><span>Python · scikit-learn · XGBoost · MLflow · Streamlit</span><span>Forecasts use assumed growth rates</span></div>', unsafe_allow_html=True)


def style_fig(fig: go.Figure, height=380, legend=True, title=None):
    fig.update_layout(
        template="none", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=height, colorway=COLORWAY,
        margin=dict(l=8, r=8, t=56 if (title or fig.layout.title.text) else 16, b=8), showlegend=legend,
        font=dict(family="Manrope, sans-serif", color=INK, size=12),
        title=dict(text=title or fig.layout.title.text, font=dict(family="Manrope, sans-serif", size=14, color=INK), x=0, xanchor="left"),
        hoverlabel=dict(bgcolor="#101714", bordercolor=MINT_DEEP, font=dict(color=INK, family="Manrope")),
        legend=dict(orientation="h", y=1.1, x=1, xanchor="right", font=dict(size=11, color=GRAPHITE)))
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="rgba(190,215,203,.28)", ticks="outside", tickcolor="rgba(190,215,203,.28)", title_font=dict(size=11, color=GRAPHITE), tickfont=dict(color=GRAPHITE))
    fig.update_yaxes(gridcolor="rgba(190,215,203,.08)", zeroline=False, linecolor="rgba(0,0,0,0)", title_font=dict(size=11, color=GRAPHITE), tickfont=dict(color=GRAPHITE))
    fig.update_annotations(font=dict(family="Manrope", color=INK, size=11))
    return fig


def annotate(fig: go.Figure, x, y, text, ax=0, ay=-42):
    fig.add_annotation(x=x, y=y, text=text, showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor=MINT_DEEP, ax=ax, ay=ay,
                       font=dict(size=11, color=INK, family="Manrope"), bgcolor="#102A22", bordercolor=MINT_DEEP, borderwidth=1, borderpad=4)
