import streamlit as st
import pandas as pd
import re
import os
import json
from fpdf import FPDF
from datetime import datetime
from engine import BacteriaIdentifier
from engine.parser_llm import evaluate_gold_tests
from training.rules_suggester import suggest_rules_from_gold, save_rule_suggestions
from training.rules_sanitizer import sanitize_rules

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(page_title="BactAI-D", layout="wide")

# -------------------------------------------------
# LOAD DATABASE
# -------------------------------------------------
@st.cache_data
def load_data(path, last_modified):
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    return df

primary_path = os.path.join("data", "bacteria_db.xlsx")
fallback_path = os.path.join("bacteria_db.xlsx")
data_path = primary_path if os.path.exists(primary_path) else fallback_path

try:
    last_modified = os.path.getmtime(data_path)
except FileNotFoundError:
    st.error(f"Database file not found at '{primary_path}' or '{fallback_path}'.")
    st.stop()

db = load_data(data_path, last_modified)
eng = BacteriaIdentifier(db)

st.sidebar.caption(
    f"📅 Database last updated: {datetime.fromtimestamp(last_modified).strftime('%Y-%m-%d %H:%M:%S')}"
)

# -------------------------------------------------
# APP TABS
# -------------------------------------------------
tab_main, tab_chat, tab_training = st.tabs(["🧫 Identify", "💬 Chat", "🧠 Training"])

# -------------------------------------------------
# TAB 1: IDENTIFY (Original UI)
# -------------------------------------------------
with tab_main:
    st.title("🧫 BactAI-D: Intelligent Bacteria Identification Assistant")
    st.markdown("Use the sidebar to input your biochemical and morphological results.")

    MORPH_FIELDS = ["Gram Stain", "Shape", "Colony Morphology", "Media Grown On", "Motility", "Capsule", "Spore Formation"]
    ENZYME_FIELDS = ["Catalase", "Oxidase", "Coagulase", "Lipase Test"]
    SUGAR_FIELDS = [
        "Glucose Fermentation", "Lactose Fermentation", "Sucrose Fermentation", "Maltose Fermentation",
        "Mannitol Fermentation", "Sorbitol Fermentation", "Xylose Fermentation", "Rhamnose Fermentation",
        "Arabinose Fermentation", "Raffinose Fermentation", "Trehalose Fermentation", "Inositol Fermentation"
    ]

    if "user_input" not in st.session_state:
        st.session_state.user_input = {}
    if "results" not in st.session_state:
        st.session_state.results = pd.DataFrame()

    def get_unique_values(field):
        vals = []
        for v in eng.db[field]:
            parts = re.split(r"[;/]", str(v))
            for p in parts:
                clean = p.strip()
                if clean and clean not in vals:
                    vals.append(clean)
        vals.sort()
        return vals

    # Sidebar Inputs
    with st.sidebar.expander("🧫 Morphological Tests", expanded=True):
        for field in MORPH_FIELDS:
            if field in ["Shape", "Colony Morphology", "Media Grown On"]:
                options = get_unique_values(field)
                selected = st.multiselect(field, options, default=[], key=field)
                st.session_state.user_input[field] = "; ".join(selected) if selected else "Unknown"
            else:
                st.session_state.user_input[field] = st.selectbox(field, ["Unknown", "Positive", "Negative", "Variable"], index=0, key=field)

    with st.sidebar.expander("🧪 Enzyme Tests", expanded=False):
        for field in ENZYME_FIELDS:
            st.session_state.user_input[field] = st.selectbox(field, ["Unknown", "Positive", "Negative", "Variable"], index=0, key=field)

    with st.sidebar.expander("🍬 Carbohydrate Fermentation Tests", expanded=False):
        for field in SUGAR_FIELDS:
            st.session_state.user_input[field] = st.selectbox(field, ["Unknown", "Positive", "Negative", "Variable"], index=0, key=field)

    with st.sidebar.expander("🧬 Other Tests", expanded=False):
        for field in db.columns:
            if field in ["Genus"] + MORPH_FIELDS + ENZYME_FIELDS + SUGAR_FIELDS:
                continue
            if field == "Haemolysis Type":
                options = get_unique_values(field)
                selected = st.multiselect(field, options, default=[], key=field)
                st.session_state.user_input[field] = "; ".join(selected) if selected else "Unknown"
            elif field == "Oxygen Requirement":
                options = get_unique_values(field)
                st.session_state.user_input[field] = st.selectbox(field, ["Unknown"] + options, index=0, key=field)
            elif field == "Growth Temperature":
                st.session_state.user_input[field] = st.text_input(field + " (°C)", "", key=field)
            else:
                st.session_state.user_input[field] = st.selectbox(field, ["Unknown", "Positive", "Negative", "Variable"], index=0, key=field)

    if st.sidebar.button("🔄 Reset All Inputs"):
        st.session_state.user_input = {k: "Unknown" for k in st.session_state.user_input}
        st.session_state.results = pd.DataFrame()
        st.rerun()

    if st.sidebar.button("🔍 Identify"):
        with st.spinner("Analyzing results..."):
            results = eng.identify(st.session_state.user_input)
            if not results:
                st.error("No matches found.")
            else:
                results_df = pd.DataFrame(
                    [
                        [
                            r.genus,
                            f"{r.confidence_percent()}%",
                            f"{r.true_confidence()}%",
                            r.reasoning_paragraph(results),
                            r.reasoning_factors.get("next_tests", ""),
                            r.extra_notes
                        ]
                        for r in results
                    ],
                    columns=["Genus", "Confidence", "True Confidence (All Tests)", "Reasoning", "Next Tests", "Extra Notes"],
                )
                st.session_state.results = results_df

    if not st.session_state.results.empty:
        st.info("Percentages based on options entered. True confidence percentage shown within each expanded result.")
        for _, row in st.session_state.results.iterrows():
            confidence_value = int(row["Confidence"].replace("%", ""))
            color = "🟢" if confidence_value >= 75 else "🟡" if confidence_value >= 50 else "🔴"
            header = f"**{row['Genus']}** — {color} {row['Confidence']}"
            with st.expander(header):
                st.markdown(f"**Reasoning:** {row['Reasoning']}")
                st.markdown(f"**Next Tests:** {row['Next Tests']}")
                st.markdown(f"**True Confidence (All Tests):** {row['True Confidence (All Tests)']}")
                if row["Extra Notes"]:
                    st.markdown(f"**Notes:** {row['Extra Notes']}")

# -------------------------------------------------
# TAB 2: CHAT UI
# -------------------------------------------------
with tab_chat:
    st.title("💬 BactAI-D Chat Assistant")
    st.markdown("Ask about microbiology, species traits, or test interpretation.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).markdown(msg["content"])

    user_msg = st.chat_input("Ask me anything microbiological...")
    if user_msg:
        st.session_state.chat_history.append({"role": "user", "content": user_msg})
        # TODO: Integrate Ollama LLM here if desired for chat responses.
        st.session_state.chat_history.append({"role": "assistant", "content": f"🧫 [Placeholder response for now] You said: {user_msg}"})
        st.rerun()

# -------------------------------------------------
# TAB 3: TRAINING
# -------------------------------------------------
with tab_training:
    st.title("🧠 Training & Gold Tests")

    col1, col2 = st.columns([2, 1])
    with col1:
        gold_path = st.text_input("Gold Tests Path", value="training/gold_tests.json")
        use_llm = st.toggle("Use LLM fallback", value=True, key="use_llm_toggle")
    with col2:
        model_name = st.text_input("Ollama model", value=os.getenv("OLLAMA_MODEL", "deepseek-coder:6.7b"))

    run_btn = st.button("▶️ Run Gold Tests")

    if run_btn:
        if not os.path.exists(gold_path):
            st.error(f"File not found: {gold_path}")
        else:
            with open(gold_path, "r", encoding="utf-8") as f:
                gold_data = json.load(f)
            with st.spinner("Evaluating... this may take a while"):
                summary = evaluate_gold_tests(gold_data, model_name=model_name, use_llm=use_llm)
                st.session_state["gold_summary"] = summary
                st.success(f"✅ Done! Overall accuracy: {summary['overall_accuracy']}%")

    if "gold_summary" in st.session_state:
        st.subheader("Gold Test Summary")
        st.json(st.session_state["gold_summary"])

        # Suggest rules button
        if st.button("💡 Suggest New Rules"):
            with open(gold_path, "r", encoding="utf-8") as f:
                gold_data = json.load(f)
            suggestions = suggest_rules_from_gold(gold_data, model_name=model_name, use_llm=use_llm)
            save_rule_suggestions(suggestions)
            st.success(f"Suggested {len(suggestions)} rule candidates. Saved to training/rule_candidates.json")

        # Sanitize rules
        if st.button("🧹 Sanitize Suggested Rules"):
            path = "training/rule_candidates.json"
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                ok, msg, cleaned = sanitize_rules(data.get("rules", []))
                st.info(msg)
                if ok:
                    st.session_state["sanitized_rules"] = cleaned
                    st.json(cleaned[:10])
            else:
                st.warning("No rule_candidates.json found. Please run Suggest New Rules first.")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center; font-size:14px;'>Created by <b>Zain</b></div>", unsafe_allow_html=True)
