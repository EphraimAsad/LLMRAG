import streamlit as st
import pandas as pd
import re
import os
import json
from datetime import datetime
from fpdf import FPDF

# --- ENGINE IMPORTS ---
from engine import BacteriaIdentifier
from engine.parser_llm import evaluate_gold_tests
from engine.parser_rules_runtime import apply_rules, merge_into_record

# --- TRAINING HELPERS ---
from training.rules_suggester import suggest_rules_from_gold, save_rule_suggestions
from training.rules_sanitizer import sanitize_rules

# --- CONFIG ---
st.set_page_config(page_title="BactAI-D", layout="wide")

# ===============================================================
# LOAD DATABASE
# ===============================================================

@st.cache_data
def load_data(path):
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    return df

data_path = "data/bacteria_db.xlsx" if os.path.exists("data/bacteria_db.xlsx") else "bacteria_db.xlsx"
if not os.path.exists(data_path):
    st.error("❌ Database file not found.")
    st.stop()

db = load_data(data_path)
eng = BacteriaIdentifier(db)

# ===============================================================
# SIDEBAR (INPUT)
# ===============================================================

st.sidebar.title("🔬 Input Test Results")

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
            c = p.strip()
            if c and c not in vals:
                vals.append(c)
    vals.sort()
    return vals

with st.sidebar.expander("🧫 Morphological Tests", expanded=True):
    for field in MORPH_FIELDS:
        if field in ["Shape", "Colony Morphology", "Media Grown On"]:
            options = get_unique_values(field)
            selected = st.multiselect(field, options, default=[], key=f"sidebar_{field}")
            st.session_state.user_input[field] = "; ".join(selected) if selected else "Unknown"
        else:
            st.session_state.user_input[field] = st.selectbox(
                field, ["Unknown", "Positive", "Negative", "Variable"], index=0, key=f"sidebar_{field}"
            )

with st.sidebar.expander("🧪 Enzyme Tests"):
    for field in ENZYME_FIELDS:
        st.session_state.user_input[field] = st.selectbox(
            field, ["Unknown", "Positive", "Negative", "Variable"], index=0, key=f"sidebar_{field}"
        )

with st.sidebar.expander("🍬 Carbohydrate Fermentation"):
    for field in SUGAR_FIELDS:
        st.session_state.user_input[field] = st.selectbox(
            field, ["Unknown", "Positive", "Negative", "Variable"], index=0, key=f"sidebar_{field}"
        )

with st.sidebar.expander("🧬 Other Tests"):
    for field in db.columns:
        if field in ["Genus"] + MORPH_FIELDS + ENZYME_FIELDS + SUGAR_FIELDS:
            continue
        if field == "Haemolysis Type":
            options = get_unique_values(field)
            selected = st.multiselect(field, options, default=[], key=f"sidebar_{field}")
            st.session_state.user_input[field] = "; ".join(selected) if selected else "Unknown"
        elif field == "Oxygen Requirement":
            options = get_unique_values(field)
            st.session_state.user_input[field] = st.selectbox(
                field, ["Unknown"] + options, index=0, key=f"sidebar_{field}"
            )
        elif field == "Growth Temperature":
            st.session_state.user_input[field] = st.text_input(field + " (°C)", "", key=f"sidebar_{field}")
        else:
            st.session_state.user_input[field] = st.selectbox(
                field, ["Unknown", "Positive", "Negative", "Variable"], index=0, key=f"sidebar_{field}"
            )

# ===============================================================
# MAIN APP BODY
# ===============================================================

tab1, tab2 = st.tabs(["💬 Identification Assistant", "🧠 Training & Evaluation"])

# ===============================================================
# TAB 1 — IDENTIFICATION
# ===============================================================

with tab1:
    st.title("🧫 BactAI-D: Intelligent Bacteria Identification")

    if st.sidebar.button("🔍 Identify"):
        with st.spinner("Analyzing results..."):
            results = eng.identify(st.session_state.user_input)
            if not results:
                st.error("No matches found.")
            else:
                df = pd.DataFrame(
                    [
                        [
                            r.genus,
                            f"{r.confidence_percent()}%",
                            f"{r.true_confidence()}%",
                            r.reasoning_paragraph(results),
                            r.reasoning_factors.get("next_tests", ""),
                            r.extra_notes,
                        ]
                        for r in results
                    ],
                    columns=["Genus", "Confidence", "True Confidence", "Reasoning", "Next Tests", "Notes"],
                )
                st.session_state.results = df

    if not st.session_state.results.empty:
        for _, row in st.session_state.results.iterrows():
            conf_val = int(row["Confidence"].replace("%", ""))
            color = "🟢" if conf_val >= 75 else "🟡" if conf_val >= 50 else "🔴"
            header = f"{color} **{row['Genus']}** — {row['Confidence']}"
            with st.expander(header):
                st.markdown(f"**Reasoning:** {row['Reasoning']}")
                st.markdown(f"**Next Tests:** {row['Next Tests']}")
                st.markdown(f"**True Confidence:** {row['True Confidence']}")
                if row["Notes"]:
                    st.markdown(f"**Notes:** {row['Notes']}")

# ===============================================================
# TAB 2 — TRAINING
# ===============================================================

with tab2:
    st.header("🧠 Model Training and Evaluation")
    st.caption("Run gold tests, generate rule suggestions, and safely push updates.")

    gold_path = st.text_input("📂 Gold Tests Path", value="training/gold_tests.json", key="gold_path")
    use_llm = st.toggle("Use LLM fallback", value=True, key="use_llm_toggle")
    model_name = st.text_input("🤖 Ollama model name", value=os.getenv("OLLAMA_MODEL", "deepseek-coder:6.7b"))

    if st.button("▶️ Run Gold Tests", key="run_gold"):
        with open(gold_path, "r", encoding="utf-8") as f:
            gold_data = json.load(f)
        with st.spinner("Evaluating model..."):
            summary = evaluate_gold_tests(gold_data, model_name=model_name, use_llm=use_llm)
            st.session_state["gold_summary"] = summary
            st.success(f"✅ Overall Accuracy: {summary['overall_accuracy']}%")
            st.json(summary["per_field_accuracy"])

    # --- Suggest New Rules ---
    if "gold_summary" in st.session_state:
        if st.button("💡 Suggest New Rules", key="suggest_rules_btn"):
            with open(gold_path, "r", encoding="utf-8") as f:
                gold_data = json.load(f)
            candidates = suggest_rules_from_gold(gold_data, model_name=model_name, use_llm=use_llm)
            save_rule_suggestions(candidates)
            st.success(f"✅ {len(candidates)} candidate rules suggested.")
            st.json(candidates[:5])

    # --- Sanitize Rules ---
    cand_path = "training/rule_candidates.json"
    if os.path.exists(cand_path):
        if st.button("🧹 Sanitize Suggested Rules", key="sanitize_rules_btn"):
            with open(cand_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            ok, msg, cleaned = sanitize_rules(data.get("rules", []))
            st.info(msg)
            if ok:
                st.session_state["sanitized_rules"] = cleaned
                st.json(cleaned[:5])

    # --- Compute weights ---
    if "gold_summary" in st.session_state:
        summary = st.session_state["gold_summary"]
        min_w = st.number_input("Min weight", value=0.4, step=0.05, key="min_w")
        max_w = st.number_input("Max weight", value=1.6, step=0.05, key="max_w")

        if st.button("⚖️ Compute Weights", key="compute_weights_btn"):
            per_field = summary.get("per_field_accuracy", {})
            # Normalize weights (inverse of error)
            weights = {f: round(max(min_w, min(max_w, (acc / 100) * 1.2)), 2) for f, acc in per_field.items()}
            os.makedirs("training", exist_ok=True)
            with open("training/field_weights.json", "w", encoding="utf-8") as f:
                json.dump(weights, f, indent=2)
            st.success("✅ Weights computed and saved to training/field_weights.json")
            st.json(weights)

    # --- Commit updates to GitHub ---
    if st.button("🚀 Commit Field Weights to GitHub", key="commit_weights_btn"):
        import subprocess
        try:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            new_branch = f"bactai-weights-update-{timestamp}"
            commit_msg = f"feat(training): update field weights {timestamp}"

            subprocess.run(["git", "checkout", "-b", new_branch], check=False)
            subprocess.run(["git", "add", "training/field_weights.json"], check=True)
            subprocess.run(["git", "commit", "-m", commit_msg], check=True)
            subprocess.run(["git", "push", "-u", "origin", new_branch], check=True)

            st.success(f"✅ Pushed {new_branch}. You can now open a PR on GitHub.")
        except subprocess.CalledProcessError as e:
            st.error(f"GitHub commit failed: {e}")

# ===============================================================
# FOOTER
# ===============================================================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center; font-size:14px;'>Created by <b>Zain</b> | © 2025 BactAI-D Project</div>",
    unsafe_allow_html=True
)
