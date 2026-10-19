"""Streamlit entry point for DriveWise Verify.

The app reads the prepared JSON library from `data/`, then exposes four
brochure-grounded workflows: ask, verify, compare, and recommend.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drivewise.assistant import DriveWiseAssistant
from drivewise.library import BrochureLibrary, load_libraries_from_data_dir
from drivewise.models import (
    AssistantResult,
    ClaimVerificationResult,
    ComparisonResult,
    RecommendationResult,
    SourceView,
)


BROCHURE_DIR = ROOT / "brochures"
DATA_DIR = ROOT / "data"
DATA_FILE = DATA_DIR / "hyundai_brochures.json"
BRAND = "Hyundai"

REQUIREMENT_PRESETS = (
    "Family use",
    "City driving",
    "Highway travel",
    "Safety",
    "Comfort",
    "Technology",
    "Space",
    "Performance",
    "EV range and charging",
)


st.set_page_config(page_title="DriveWise Verify", page_icon="🚘", layout="wide")

st.markdown(
    """
    <style>
    :root { --dw-ink:#182d39; --dw-muted:#526774; --dw-teal:#087582; --dw-line:#8196a3; color-scheme:light; }
    html, body, [class*="css"] { font-family: 'Segoe UI', Arial, sans-serif; }
    .stApp { background:#f3f7fa; color:var(--dw-ink); }
    [data-testid="stHeader"] { background:transparent; }
    .block-container { max-width:1200px; padding:2.5rem 3rem 4rem; }
    h1,h2,h3 { color:var(--dw-ink); letter-spacing:-.035em; }
    h2 { font-size:1.5rem !important; }
    [data-testid="stCaptionContainer"] { color:var(--dw-muted); }
    [data-testid="stSidebar"] { background:#102c3a; border-right:0; }
    [data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,[data-testid="stSidebar"] label { color:#e4eff4; }
    [data-testid="stSidebar"] [data-testid="stMetric"] { background:#1c3b49; border-color:#2c4c59; }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] { color:#8ce1dc; }
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] p { color:#b8ced8; }
    [data-testid="stMetric"] { background:white; border:1px solid var(--dw-line); padding:1rem; border-radius:14px; }
    [data-testid="stMetricValue"] { color:var(--dw-teal); font-size:1.7rem; }
    [data-testid="stVerticalBlockBorderWrapper"] > div { border-radius:20px !important; }
    [data-testid="stTextArea"] textarea { background:#fbfdfe; color:#182d39; border-radius:12px; line-height:1.6; }
    [data-baseweb="select"] > div { background:#fbfdfe; color:#182d39; border-radius:10px; }
    [data-testid="stTextArea"] textarea:focus { border-color:#087e8b; box-shadow:0 0 0 3px #087e8b18; }
    .stButton > button { border-radius:10px; transition:background .18s, box-shadow .18s; }
    .stButton > button[kind="primary"] { background:#087e8b; color:white; border:0; min-height:3rem; font-weight:600; }
    .stButton > button[kind="primary"]:hover { background:#056571; box-shadow:0 5px 16px #087e8b25; }
    [data-testid="stRadio"] > div { gap:.6rem; }
    [data-testid="stRadio"] label { background:white; border:1px solid var(--dw-line); border-radius:12px; padding:.7rem 1rem; }
    [data-testid="stRadio"] label:has(input:checked) { background:#e2f3f3; border-color:#087e8b; }
    .dw-brand { font-size:1.5rem; font-weight:750; color:white; margin-bottom:.3rem; }
    .dw-sidebar-note { color:#adc6d2; font-size:.9rem; line-height:1.7; margin-bottom:1.8rem; }
    .dw-hero { position:relative; overflow:hidden; background:linear-gradient(115deg,#102e40,#155369); border-radius:24px; padding:2.8rem; margin-bottom:1.7rem; }
    .dw-hero:after { content:""; position:absolute; width:260px; height:260px; border:45px solid #ffffff08; border-radius:50%; right:-55px; top:-95px; pointer-events:none; }
    .dw-eyebrow { color:#8ce1dc; font-size:.72rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; margin-bottom:1rem; }
    .dw-hero h1 { color:white !important; font-size:clamp(2rem,4vw,3.2rem); line-height:1.12; margin:0 0 1rem; max-width:650px; }
    .dw-hero p { color:#c7dce7; max-width:580px; font-size:1.05rem; line-height:1.7; }
    .dw-badge { display:inline-block; color:#d3eeef; background:#ffffff0d; border:1px solid #ffffff20; border-radius:30px; padding:.35rem .75rem; font-size:.75rem; margin:.6rem .4rem 0 0; }
    .dw-section-kicker { color:#526774; font-size:.72rem; letter-spacing:.13em; font-weight:700; text-transform:uppercase; margin:.5rem 0 .8rem; }
    .dw-footer { text-align:center; color:#526774; font-size:.8rem; margin-top:2rem; }
    /* Keep native controls and their icons consistent with the page palette. */
    [data-testid="stMain"] { color:var(--dw-ink); }
    [data-testid="stWidgetLabel"] p,
    [data-testid="stRadio"] label p { color:var(--dw-ink); }
    [data-testid="stCaptionContainer"] p { color:var(--dw-muted); }
    [data-testid="stTextArea"] textarea::placeholder { color:#526774; opacity:1; }
    [data-testid="stTextArea"] textarea,
    [data-baseweb="select"] > div { border:1px solid #8196a3; }
    [data-baseweb="select"] svg { color:#244a5d; fill:currentColor; }
    [data-baseweb="tag"] { background:#dceff0; color:#164e56; }
    [data-baseweb="tag"] span { color:#164e56; }
    [data-baseweb="tag"] svg { color:#164e56; fill:currentColor; }
    [data-baseweb="popover"], [role="listbox"] { background:#ffffff; color:#182d39; }
    [role="option"] { color:#182d39; }
    [role="option"][aria-selected="true"] { background:#dceff0; color:#164e56; }
    [data-testid="stExpander"] details { background:#ffffff; color:#182d39; border-color:#8196a3; }
    [data-testid="stExpander"] summary { color:#182d39; }
    [data-testid="stExpander"] summary svg { color:inherit; fill:currentColor; }
    .stButton > button[kind="primary"],
    .stButton > button[kind="primary"] p { background:#087582; color:#ffffff; }
    .stButton > button[kind="primary"]:hover p { background:transparent; }
    button:focus-visible, summary:focus-visible,
    [data-baseweb="select"]:focus-within { outline:3px solid #087582; outline-offset:3px; }
    [data-testid="stHeader"] button { color:#244a5d; }
    [data-testid="stSidebar"] [data-testid="stExpander"] details { background:#1c3b49; border-color:#7897a8; }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary,
    [data-testid="stSidebar"] button { color:#e4eff4; }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color:#b8ced8; }
    @media(max-width:768px) { .block-container { padding:1.5rem 1rem 3rem; } .dw-hero { padding:1.8rem 1.4rem; } [data-testid="stRadio"] label { padding:.5rem .6rem; } }
    </style>
    """, unsafe_allow_html=True,
)

@st.cache_data(show_spinner=False)
def cached_base_library(data_signature: tuple[tuple[str, float, int], ...]) -> BrochureLibrary:
    return load_libraries_from_data_dir(DATA_DIR)


def data_signature() -> tuple[tuple[str, float, int], ...]:
    """Give Streamlit a simple cache key for the generated JSON files."""
    return tuple(
        (path.name, path.stat().st_mtime, path.stat().st_size)
        for path in sorted(DATA_DIR.glob("*.json"))
    )


def pdf_count() -> int:
    """Count source brochures so the sidebar shows the dataset status."""
    return len(list(BROCHURE_DIR.glob("*.pdf")))


def source_records(sources: tuple[SourceView, ...]) -> list[dict[str, str | int | float]]:
    return [
        {
            "brand": source.brand,
            "model": source.model,
            "section": source.section,
            "page": source.page,
            "brochure": source.brochure_name,
            "score": round(source.score, 3),
        }
        for source in sources
    ]


def render_sources(sources: tuple[SourceView, ...], heading: str = "Sources") -> None:
    st.subheader(heading)
    if not sources:
        st.info("No supporting brochure section was strong enough for this response.")
        return

    st.dataframe(
        source_records(sources),
        use_container_width=True,
        hide_index=True,
    )


def render_context(sources: tuple[SourceView, ...], label: str = "Retrieved context") -> None:
    with st.expander(label):
        if not sources:
            st.write("No retrieved context.")
            return
        for source in sources:
            st.markdown(
                f"**{source.model} | {source.section}** | page {source.page} | score {source.score:.3f}"
            )
            st.write(source.preview)


def render_quality(result: AssistantResult | ClaimVerificationResult | ComparisonResult) -> None:
    st.caption("Evidence quality · These scores are local estimates of brochure support, not independent accuracy ratings.")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Response", f"{result.response_ms} ms")
    metric_cols[1].metric("Faithfulness", f"{result.evaluation.faithfulness:.2f}")
    metric_cols[2].metric("Coverage", f"{result.evaluation.answer_coverage:.2f}")
    metric_cols[3].metric("Relevance", f"{result.evaluation.context_relevance:.2f}")


def run_ask_mode(models: list[str], assistant: DriveWiseAssistant) -> None:
    top_cols = st.columns([0.78, 0.22])
    with top_cols[0]:
        model = st.selectbox("Model", models, key="ask_model")
    with top_cols[1]:
        st.metric("Brochure passages", str(len(library.filter_chunks(BRAND, model))))

    question = st.text_area(
        "Ask about the selected brochure",
        value="",
        height=96,
        placeholder="What safety and comfort features are available?",
        key="ask_question",
    )

    if st.button("Find my answer →", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Enter a question to get started.")
            return
        result = assistant.answer(brand=BRAND, model=model, question=question)
        st.subheader("Answer")
        st.write(result.answer)
        render_quality(result)
        render_sources(result.sources)
        render_context(result.sources)
        with st.expander("Quality signals"):
            st.json(result.evaluation.as_dict())


def run_verify_mode(models: list[str], assistant: DriveWiseAssistant) -> None:
    top_cols = st.columns([0.78, 0.22])
    with top_cols[0]:
        model = st.selectbox("Model", models, key="verify_model")
    with top_cols[1]:
        st.metric("Brochure passages", str(len(library.filter_chunks(BRAND, model))))

    claim = st.text_area(
        "Automotive claim",
        value="",
        height=96,
        placeholder="Example: This model has six airbags.",
        key="claim_text",
    )

    if st.button("Check this claim →", type="primary", use_container_width=True):
        if not claim.strip():
            st.warning("Enter the claim you would like to check.")
            return
        result = assistant.verify_claim(brand=BRAND, model=model, claim=claim)
        verdict_cols = st.columns(4)
        verdict_cols[0].metric("Verdict", result.verdict)
        verdict_cols[1].metric("Confidence", f"{result.confidence:.2f}")
        verdict_cols[2].metric("Sources", str(len(result.sources)))
        verdict_cols[3].metric("Response", f"{result.response_ms} ms")

        st.subheader("Explanation")
        st.write(result.explanation)

        st.subheader("Evidence")
        if result.evidence:
            for sentence in result.evidence:
                st.markdown(f"- {sentence}")
        else:
            st.info("No sentence-level brochure evidence was strong enough to show.")

        render_quality(result)
        render_sources(result.sources)
        render_context(result.sources)


def run_compare_mode(models: list[str], assistant: DriveWiseAssistant) -> None:
    left, right = st.columns(2)
    with left:
        left_model = st.selectbox("Model A", models, key="compare_left")
    with right:
        default_index = 1 if len(models) > 1 else 0
        right_model = st.selectbox("Model B", models, index=default_index, key="compare_right")

    question = st.text_area(
        "Comparison focus",
        value="Compare their safety and comfort features.",
        height=96,
        placeholder="Example: Compare dimensions, safety, comfort, and technology.",
        key="compare_question",
    )

    if st.button("Compare these cars →", type="primary", use_container_width=True):
        if left_model == right_model:
            st.warning("Choose two different models to compare.")
            return
        if not question.strip():
            st.warning("Tell us which features you want to compare.")
            return
        result = assistant.compare_models(
            brand=BRAND,
            left_model=left_model,
            right_model=right_model,
            question=question,
        )
        st.subheader("Brochure-Supported Comparison")
        st.dataframe(result.table_records(), use_container_width=True, hide_index=True)
        render_quality(result)

        source_tabs = st.tabs([left_model, right_model])
        with source_tabs[0]:
            render_sources(result.left_sources, heading=f"{left_model} sources")
            render_context(result.left_sources, label=f"{left_model} retrieved context")
        with source_tabs[1]:
            render_sources(result.right_sources, heading=f"{right_model} sources")
            render_context(result.right_sources, label=f"{right_model} retrieved context")


def run_recommend_mode(assistant: DriveWiseAssistant) -> None:
    selected_needs = st.multiselect(
        "Requirement focus",
        REQUIREMENT_PRESETS,
        default=["Family use", "Highway travel", "Safety"],
    )
    custom_requirements = st.text_area(
        "Requirement details",
        value="I need a car mainly for family highway trips with strong safety and comfort.",
        height=104,
        placeholder="Example: I want a city car with good mileage, easy parking, and modern technology.",
    )
    top_n = st.slider("Recommendations to show", min_value=1, max_value=5, value=3)
    requirements = " ".join([custom_requirements, " ".join(selected_needs)]).strip()

    if st.button("Find my matches →", type="primary", use_container_width=True):
        if not requirements:
            st.warning("Select a priority or describe what you need.")
            return
        result = assistant.recommend(brand=BRAND, requirements=requirements, top_n=top_n)
        if not result.candidates:
            st.info("No brochure-backed recommendation could be made for those requirements.")
            return

        st.subheader("Evidence-Based Recommendations")
        st.dataframe(result.table_records(), use_container_width=True, hide_index=True)
        st.metric("Response", f"{result.response_ms} ms")

        for index, candidate in enumerate(result.candidates, start=1):
            with st.expander(f"{index}. {candidate.model} | fit {candidate.fit_score:.2f}", expanded=index == 1):
                st.write(candidate.explanation)
                st.caption("Evidence")
                if candidate.evidence:
                    for sentence in candidate.evidence:
                        st.markdown(f"- {sentence}")
                else:
                    st.write("No concise evidence sentence was extracted.")
                render_sources(candidate.sources, heading="Sources")
                st.caption("Supporting brochure passages")
                for source in candidate.sources:
                    st.markdown(f"**{source.model} · {source.section} · page {source.page}**")
                    st.write(source.preview)


library = cached_base_library(data_signature())
assistant = DriveWiseAssistant(library)

if not library.chunks:
    st.error("No brochure data is available. Run `python scripts/ingest_hyundai_brochures.py`.")
    st.stop()

models = library.models_for_brand(BRAND)

with st.sidebar:
    st.markdown('<div class="dw-brand">◈ DriveWise</div><div class="dw-sidebar-note">A little clarity for your next big drive.</div>', unsafe_allow_html=True)
    st.caption("YOUR BROCHURE LIBRARY")
    st.metric("Hyundai models", str(len(models)))
    st.metric("Official brochures", str(pdf_count()))
    st.divider()
    st.markdown("### Every answer has a source")
    st.write("Explore features, check claims, and compare cars using the brochures in your library.")
    with st.expander("Library details"):
        st.caption(f"{len(library.chunks):,} searchable brochure passages")
        st.caption("Brand: Hyundai")
        if DATA_FILE.exists():
            st.caption(f"Library size: {DATA_FILE.stat().st_size / 1024:.1f} KB")
    st.divider()
    st.caption("Features may vary by variant and brochure edition. Check the cited pages for details.")

st.markdown(
    """
    <div class="dw-hero">
        <div class="dw-eyebrow">Your brochure-powered car companion</div>
        <h1>Your next car.<br>A clearer choice.</h1>
        <p>Get to know your Hyundai, check the details, and find what fits your life. All backed by official brochure evidence.</p>
        <span class="dw-badge">Hyundai library</span>
        <span class="dw-badge">Traceable sources</span>
        <span class="dw-badge">Four ways to explore</span>
    </div>
    """, unsafe_allow_html=True,
)
st.markdown('<div class="dw-section-kicker">What would you like to do?</div>', unsafe_allow_html=True)
mode = st.radio(
    "Interaction mode",
    ["Ask", "Verify Claim", "Compare Models", "Recommend"],
    format_func=lambda value: {"Ask": "Ask a question", "Verify Claim": "Check a claim", "Compare Models": "Compare cars", "Recommend": "Find my match"}[value],
    horizontal=True,
    label_visibility="collapsed",
)

mode_details = {
    "Ask": ("Explore the details", "Choose a model and ask about the features that matter to you."),
    "Verify Claim": ("Get the facts", "Check something you heard against the selected model's brochure."),
    "Compare Models": ("Find the differences", "Pick two cars and tell us what you want to compare."),
    "Recommend": ("A car that fits your life", "Share your priorities to discover matches from the brochure library."),
}
with st.container(border=True):
    title, description = mode_details[mode]
    st.subheader(title)
    st.caption(description)
    if mode == "Ask":
        run_ask_mode(models, assistant)
    elif mode == "Verify Claim":
        run_verify_mode(models, assistant)
    elif mode == "Compare Models":
        run_compare_mode(models, assistant)
    else:
        run_recommend_mode(assistant)

st.markdown('<div class="dw-footer">DriveWise Verify · Explore with confidence. Follow the evidence.</div>', unsafe_allow_html=True)
