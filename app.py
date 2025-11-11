import streamlit as st
import pandas as pd
import os
import json
from datetime import datetime
from fpdf import FPDF

from engine import BacteriaIdentifier
from engine.parser_llm import evaluate_gold_tests
from training.rules_suggester import suggest_rules_from_gold, save_rule_suggestions
from training.rules_sanitizer import sanitize_rules

# --- CONFIG ---
st.set_page_config(page_title="BactAI-D Assistant", layout="wide")

# --- LOAD DATA ---
@st.cache_data
def load_data(path):
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def get_db():
    primary_path = os.path.join("data", "bacteria_db.xlsx")
    fallback_path = os.path.join("bacteria_db.xlsx")
    data_path = primary_path if os.path.exists(primary_path) else fallback_path
    if not os.path.exists(data_path):
        st.error(f"Database file not found at {primary_path} or {fallback_path}")
        st.stop()
    return load_data(data_path)


# --- PAGE NAV ---
tabs = st.tabs(["🧫 Identification", "📊 Training"])

# --- IDENTIFICATION TAB ---
with tabs[0]:
    db = get_db()
    eng = BacteriaIdentifier(db)

    st.title("🧫 BactAI-D: Intelligent Bacteria Identification Assistant")
    st.markdown("Use the sidebar to input your biochemical and morphological results.")

    MORPH_FIELDS = ["Gram Stain", "Shape", "Colony Morphology", "Media Grown On", "Motility", "Capsule", "Spore Formation"]
    ENZYME_FIELDS = ["Catalase", "Oxidase", "Coagulase", "Lipase Test"]
    SUGAR_FIELDS = [
        "Glucose Fermentation", "Lactose Fermentation", "Sucrose Fermentation",
        "Maltose Fermentation", "Mannitol Fermentation", "Sorbitol Fermentation",
        "Xylose Fermentation", "Rhamnose Fermentation", "Arabinose Fermentation",
        "Raffinose Fermentation", "Trehalose Fermentation", "Inositol Fermentation"
    ]

    if "user_input" not in st.session_state:
        st.session_state.user_input = {}
    if "results" not in st.session_state:
        st.session_state.results = pd.DataFrame()

    # Sidebar inputs
    st.sidebar.header("🔬 Input Test Results")

    def get_unique_values(field):
        vals = []
        for v in eng.db[field]:
            for p in str(v).split(";"):
                p = p.strip()
                if p and p not in vals:
                    vals.append(p)
        vals.sort()
        return vals

    with st.sidebar.expander("🧫 Morphological Tests", expanded=True):
        for field in MORPH_FIELDS:
            if field in ["Shape", "Colony Morphology", "Media Grown On"]:
                opts = get_unique_values(field)
                val = st.multiselect(field, opts, default=[])
                st.session_state.user_input[field] = "; ".join(val) if val else "Unknown"
            else:
                st.session_state.user_input[field] = st.selectbox(field, ["Unknown", "Positive", "Negative", "Variable"])

    with st.sidebar.expander("🧪 Enzyme Tests", expanded=False):
        for field in ENZYME_FIELDS:
            st.session_state.user_input[field] = st.selectbox(field, ["Unknown", "Positive", "Negative", "Variable"])

    with st.sidebar.expander("🍬 Sugar Fermentation", expanded=False):
        for field in SUGAR_FIELDS:
            st.session_state.user_input[field] = st.selectbox(field, ["Unknown", "Positive", "Negative", "Variable"])

    with st.sidebar.expander("🧬 Other Tests", expanded=False):
        for field in db.columns:
            if field in ["Genus"] + MORPH_FIELDS + ENZYME_FIELDS + SUGAR_FIELDS:
                continue
            if field == "Haemolysis Type":
                opts = get_unique_values(field)
                val = st.multiselect(field, opts, default=[])
                st.session_state.user_input[field] = "; ".join(val) if val else "Unknown"
            elif field == "Oxygen Requirement":
                opts = get_unique_values(field)
                st.session_state.user_input[field] = st.selectbox(field, ["Unknown"] + opts, index=0)
            elif field == "Growth Temperature":
                st.session_state.user_input[field] = st.text_input(field + " (°C)", "")
            else:
                st.session_state.user_input[field] = st.selectbox(field, ["Unknown", "Positive", "Negative", "Variable"])

    if st.sidebar.button("🔍 Identify"):
        with st.spinner("Analyzing results..."):
            results = eng.identify(st.session_state.user_input)
            if not results:
                st.error("No matches found.")
            else:
                rows = []
                for r in results:
                    rows.append([
                        r.genus,
                        f"{r.confidence_percent()}%",
                        f"{r.true_confidence()}%",
                        r.reasoning_paragraph(results),
                        r.reasoning_factors.get("next_tests", ""),
                        r.extra_notes
                    ])
                st.session_state.results = pd.DataFrame(
                    rows,
                    columns=["Genus", "Confidence", "True Confidence", "Reasoning", "Next Tests", "Extra Notes"]
                )

    if not st.session_state.results.empty:
        st.info("Percentages are based on the entered tests. Expand each result for reasoning.")
        for _, row in st.session_state.results.iterrows():
            conf_val = int(row["Confidence"].replace("%", ""))
            color = "🟢" if conf_val >= 75 else "🟡" if conf_val >= 50 else "🔴"
            with st.expander(f"**{row['Genus']}** — {color} {row['Confidence']}"):
                st.markdown(f"**Reasoning:** {row['Reasoning']}")
                st.markdown(f"**Next Tests:** {row['Next Tests']}")
                st.markdown(f"**True Confidence:** {row['True Confidence']}")
                if row["Extra Notes"]:
                    st.markdown(f"**Notes:** {row['Extra Notes']}")

# ------------------------------
# TRAINING TAB
# ------------------------------
with tabs[1]:
    st.header("📊 Model Training & Evaluation")

    col1, col2 = st.columns([2, 1])
    with col1:
        gold_path = st.text_input("Gold Tests Path", value="training/gold_tests.json", key="gold_path_inp")
        use_llm = st.toggle("Use LLM fallback", value=True, key="use_llm_tog")
    with col2:
        model_name = st.text_input("Ollama Model", value=os.getenv("OLLAMA_MODEL", "deepseek-coder:6.7b"), key="model_inp")

    run_btn = st.button("▶️ Run Gold Tests")

    if run_btn:
        if not os.path.exists(gold_path):
            st.error(f"Gold test file not found: {gold_path}")
        else:
            with open(gold_path, "r", encoding="utf-8") as f:
                gold_data = json.load(f)
            with st.spinner("Running gold tests..."):
                summary = evaluate_gold_tests(gold_data, model_name=model_name, use_llm=use_llm)
                st.session_state["gold_summary"] = summary
            st.success(f"✅ Completed. Overall accuracy: {summary['overall_accuracy']}%")
            st.json(summary["per_field_accuracy"])

    # Rule suggestion section
    if "gold_summary" in st.session_state and st.session_state["gold_summary"]:
        if st.button("💡 Suggest New Rules"):
            gold_data = json.load(open("training/gold_tests.json", "r", encoding="utf-8"))
            with st.spinner("Generating candidate rules..."):
                candidates = suggest_rules_from_gold(gold_data, model_name=model_name, use_llm=use_llm)
                save_rule_suggestions(candidates)
            st.success(f"✅ Suggested {len(candidates)} rule candidates. Saved to training/rule_candidates.json")

        if st.button("🧹 Sanitize Suggested Rules"):
            if not os.path.exists("training/rule_candidates.json"):
                st.error("No rule_candidates.json found. Run 'Suggest New Rules' first.")
            else:
                with open("training/rule_candidates.json", "r", encoding="utf-8") as f:
                    data = json.load(f)
                ok, msg, cleaned = sanitize_rules(data.get("rules", []))
                st.info(msg)
                if ok:
                    st.session_state["sanitized_rules"] = cleaned
                    with open("training/sanitized_rules.json", "w", encoding="utf-8") as f:
                        json.dump({"timestamp": datetime.now().isoformat(), "rules": cleaned}, f, indent=2)
                    st.success("✅ Saved sanitized rules to training/sanitized_rules.json")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center; font-size:14px;'>Created by <b>Zain</b></div>", unsafe_allow_html=True)
