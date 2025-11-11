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
from training.github_sync import commit_to_github
from training.weights_logic import compute_weights_from_accuracy, sanitize_weights, save_weights_file

# --- CONFIG ---
st.set_page_config(page_title="BactAI-D Assistant", layout="wide")

# --- LOAD DATA ---
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

# --- PAGE TABS ---
tabs = st.tabs(["💬 Chat", "🧫 BactAI-D Original", "🧠 Training"])

# -------------------------------------------------------------
# TAB 1 — Chat (placeholder for future LLM dialogue interface)
# -------------------------------------------------------------
with tabs[0]:
    st.title("💬 Chat Mode (Coming Soon)")
    st.write("This mode will allow direct dialogue with the microbiology LLM once integrated.")

# -------------------------------------------------------------
# TAB 2 — Original BactAI-D Engine
# -------------------------------------------------------------
with tabs[1]:
    st.title("🧫 BactAI-D: Intelligent Bacteria Identification Assistant")
    st.markdown("Use the sidebar to input your biochemical and morphological results.")

    MORPH_FIELDS = [
        "Gram Stain", "Shape", "Colony Morphology", "Media Grown On",
        "Motility", "Capsule", "Spore Formation"
    ]
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

    # Sidebar inputs
    st.sidebar.markdown(
        """
        <div style='background-color:#1565C0; padding:12px; border-radius:10px;'>
            <h3 style='text-align:center; color:white; margin:0;'>🔬 Input Test Results</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.sidebar.expander("🧫 Morphological Tests", expanded=True):
        for field in MORPH_FIELDS:
            if field in ["Shape", "Colony Morphology", "Media Grown On"]:
                options = get_unique_values(field)
                selected = st.multiselect(field, options, default=[], key=field)
                st.session_state.user_input[field] = "; ".join(selected) if selected else "Unknown"
            else:
                st.session_state.user_input[field] = st.selectbox(
                    field, ["Unknown", "Positive", "Negative", "Variable"], index=0, key=field
                )

    with st.sidebar.expander("🧪 Enzyme Tests", expanded=False):
        for field in ENZYME_FIELDS:
            st.session_state.user_input[field] = st.selectbox(
                field, ["Unknown", "Positive", "Negative", "Variable"], index=0, key=field
            )

    with st.sidebar.expander("🍬 Carbohydrate Fermentation Tests", expanded=False):
        for field in SUGAR_FIELDS:
            st.session_state.user_input[field] = st.selectbox(
                field, ["Unknown", "Positive", "Negative", "Variable"], index=0, key=field
            )

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
                st.session_state.user_input[field] = st.selectbox(
                    field, ["Unknown"] + options, index=0, key=field
                )
            elif field == "Growth Temperature":
                st.session_state.user_input[field] = st.text_input(field + " (°C)", "", key=field)
            else:
                st.session_state.user_input[field] = st.selectbox(
                    field, ["Unknown", "Positive", "Negative", "Variable"], index=0, key=field
                )

    if st.sidebar.button("🔄 Reset All Inputs"):
        st.session_state["reset_trigger"] = True
        st.rerun()

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
                    columns=["Genus", "Confidence", "True Confidence (All Tests)", "Reasoning", "Next Tests", "Extra Notes"],
                )
                st.session_state.results = df

    if not st.session_state.results.empty:
        st.info("Percentages based upon options entered.")
        for _, row in st.session_state.results.iterrows():
            conf_val = int(row["Confidence"].replace("%", ""))
            color = "🟢" if conf_val >= 75 else "🟡" if conf_val >= 50 else "🔴"
            header = f"**{row['Genus']}** — {color} {row['Confidence']}"
            with st.expander(header):
                st.markdown(f"**Reasoning:** {row['Reasoning']}")
                st.markdown(f"**Next Tests:** {row['Next Tests']}")
                st.markdown(f"**True Confidence:** {row['True Confidence (All Tests)']}")
                if row["Extra Notes"]:
                    st.markdown(f"**Notes:** {row['Extra Notes']}")

# -------------------------------------------------------------
# TAB 3 — Training & Continuous Learning
# -------------------------------------------------------------
with tabs[2]:
    st.title("🧠 Training Dashboard")

    st.markdown("Use this tab to run gold tests, compute field weights, suggest new rules, and sync updates to GitHub.")

    # --- Gold Test Runner ---
    st.subheader("📘 Gold Standard Evaluation")
    col1, col2 = st.columns([2, 1])
    with col1:
        gold_path = st.text_input("Gold Tests Path", value="training/gold_tests.json", key="gold_path_input")
        use_llm = st.toggle("Use LLM Fallback", value=True, key="use_llm_toggle")
    with col2:
        model_name = st.text_input("Ollama Model", value=os.getenv("OLLAMA_MODEL", "deepseek-coder:6.7b"), key="model_name_input")

    if st.button("▶️ Run Gold Tests", key="run_gold_tests_btn"):
        try:
            with open(gold_path, "r", encoding="utf-8") as f:
                gold_data = json.load(f)
            summary = evaluate_gold_tests(gold_data, model_name=model_name, use_llm=use_llm)
            st.session_state["gold_summary"] = summary
            st.success(f"✅ Overall accuracy: {summary['overall_accuracy']}%")
            st.json(summary["per_field_accuracy"])
        except Exception as e:
            st.error(f"Error: {e}")

    # --- Compute Field Weights ---
    st.subheader("⚖️ Compute Field Weights from Accuracy")
    min_w = st.number_input("Min weight", value=0.4, step=0.05, key="min_w_input")
    max_w = st.number_input("Max weight", value=1.6, step=0.05, key="max_w_input")

    if st.button("⚙️ Compute Weights", key="compute_weights_btn"):
        summary = st.session_state.get("gold_summary")
        if not summary:
            st.warning("Run gold tests first.")
        else:
            weights = compute_weights_from_accuracy(summary["per_field_accuracy"], min_w, max_w)
            ok, msg = sanitize_weights(weights, min_w=min_w, max_w=max_w)
            if ok:
                save_weights_file(weights)
                st.success("✅ Field weights computed and saved.")
                st.json(weights)
            else:
                st.error(f"Sanitization failed: {msg}")

    # --- GitHub Sync for Weights ---
    st.subheader("🌐 Commit to GitHub")
    repo_url = st.text_input("GitHub Repo URL", value=os.getenv("GITHUB_REPO", ""), key="gh_repo_inp")
    token = st.text_input("GitHub Token", type="password", key="gh_token_inp")
    base_branch = st.text_input("Base Branch", value="main", key="gh_base_branch_inp")
    new_branch_name = f"bactai-weights-update-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    if st.button("⬆️ Commit Field Weights", key="commit_weights_btn"):
        try:
            ok, msg = commit_to_github(
                repo_url=repo_url,
                token=token,
                base_branch=base_branch,
                new_branch=new_branch_name,
                file_path="training/field_weights.json",
                commit_message=f"feat(training): update field weights ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})"
            )
            if ok:
                st.success("✅ Field weights committed successfully!")
            else:
                st.error(f"GitHub commit failed: {msg}")
        except Exception as e:
            st.error(f"GitHub commit error: {e}")

    # --- Rule Learning Section ---
    st.subheader("🧩 Suggest New Rules from Gold Tests")

    if st.button("🔍 Suggest New Rules", key="suggest_rules_btn"):
        try:
            with open(gold_path, "r", encoding="utf-8") as f:
                gold_data = json.load(f)
            candidates = suggest_rules_from_gold(gold_data, model_name=model_name, use_llm=use_llm)
            save_rule_suggestions(candidates)
            st.session_state["rule_candidates"] = candidates
            st.success(f"Generated {len(candidates)} candidate rules.")
        except Exception as e:
            st.error(f"Error suggesting rules: {e}")

    if st.button("🧹 Sanitize Suggested Rules", key="sanitize_rules_btn"):
        try:
            path = "training/rule_candidates.json"
            if not os.path.exists(path):
                st.warning("No candidate rules found. Run 'Suggest Rules' first.")
            else:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                ok, msg, cleaned = sanitize_rules(data.get("rules", []))
                st.info(msg)
                if ok:
                    st.session_state["sanitized_rules"] = cleaned
                    st.success(f"{len(cleaned)} sanitized rules ready for merging.")
        except Exception as e:
            st.error(f"Sanitization failed: {e}")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; font-size:14px;'>Created by <b>Zain</b></div>", unsafe_allow_html=True)
