import streamlit as st
import pandas as pd
import os
import re
import json
from fpdf import FPDF
from datetime import datetime
from engine import BacteriaIdentifier
from engine.parser_llm import evaluate_gold_tests
from training.rules_suggester import suggest_rules_from_gold, save_rule_suggestions
from training.rules_sanitizer import sanitize_rules
from training.github_sync import commit_to_github  # assuming you already have this
from engine.parser_rules_runtime import apply_rules, merge_into_record


# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="BactAI-D Assistant", layout="wide")
TABS = ["🔬 Identify", "💬 Chat / Analysis", "🧠 Training"]
selected_tab = st.sidebar.radio("Navigation", TABS)

# -----------------------------
# LOAD DATABASE
# -----------------------------
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

# -----------------------------
# SESSION STATE
# -----------------------------
if "user_input" not in st.session_state:
    st.session_state.user_input = {}
if "results" not in st.session_state:
    st.session_state.results = pd.DataFrame()
if "gold_summary" not in st.session_state:
    st.session_state.gold_summary = None
if "sanitized_rules" not in st.session_state:
    st.session_state.sanitized_rules = []

# -----------------------------
# TAB 1 — IDENTIFICATION
# -----------------------------
if selected_tab == "🔬 Identify":
    st.title("🧫 BactAI-D: Intelligent Bacteria Identification Assistant")
    st.markdown("Use the sidebar to input your biochemical and morphological results.")

    MORPH_FIELDS = ["Gram Stain", "Shape", "Colony Morphology", "Media Grown On", "Motility", "Capsule", "Spore Formation"]
    ENZYME_FIELDS = ["Catalase", "Oxidase", "Coagulase", "Lipase Test"]
    SUGAR_FIELDS = [
        "Glucose Fermentation", "Lactose Fermentation", "Sucrose Fermentation", "Maltose Fermentation",
        "Mannitol Fermentation", "Sorbitol Fermentation", "Xylose Fermentation", "Rhamnose Fermentation",
        "Arabinose Fermentation", "Raffinose Fermentation", "Trehalose Fermentation", "Inositol Fermentation"
    ]

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

    # --------------------------
    # Sidebar Inputs
    # --------------------------
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
        for key in st.session_state.user_input.keys():
            st.session_state.user_input[key] = "Unknown"
        st.rerun()

    if st.sidebar.button("🔍 Identify"):
        with st.spinner("Analyzing results..."):
            results = eng.identify(st.session_state.user_input)
            if not results:
                st.error("No matches found.")
            else:
                results = pd.DataFrame(
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
                st.session_state.results = results

    # Display results
    if not st.session_state.results.empty:
        st.info("Percentages based on tests entered.")
        for _, row in st.session_state.results.iterrows():
            confidence_value = int(row["Confidence"].replace("%", ""))
            color = "🟢" if confidence_value >= 75 else "🟡" if confidence_value >= 50 else "🔴"
            header = f"**{row['Genus']}** — {color} {row['Confidence']}"
            with st.expander(header):
                st.markdown(f"**Reasoning:** {row['Reasoning']}")
                st.markdown(f"**Top 3 Next Tests:** {row['Next Tests']}")
                st.markdown(f"**True Confidence:** {row['True Confidence (All Tests)']}")

# -----------------------------
# TAB 2 — CHAT / ANALYSIS
# -----------------------------
elif selected_tab == "💬 Chat / Analysis":
    st.title("💬 BactAI-D Chat / Discussion")
    st.markdown("Use this area for conversational analysis, test interpretation, or explanations.")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    user_msg = st.text_input("Enter a message or question:")
    if st.button("Send"):
        st.session_state.chat_history.append({"role": "user", "content": user_msg})
        st.session_state.chat_history.append(
            {"role": "assistant", "content": "This is where your AI chat logic will go."}
        )
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"🧍 **You:** {msg['content']}")
        else:
            st.markdown(f"🤖 **BactAI-D:** {msg['content']}")

# -----------------------------
# TAB 3 — TRAINING
# -----------------------------
elif selected_tab == "🧠 Training":
    st.title("🧠 Training & Model Improvement")
    st.markdown("Use this section to evaluate gold standards, compute weights, and manage learned rules.")

    gold_path = st.text_input("Gold tests path", value="training/gold_tests.json", key="gold_path_inp")
    use_llm = st.toggle("Use LLM fallback", value=True, key="use_llm_toggle")
    model_name = st.text_input("Ollama model", value=os.getenv("OLLAMA_MODEL", "deepseek-coder:6.7b"), key="ollama_model_inp")

    if st.button("▶️ Run Gold Tests", key="run_gold"):
        try:
            with open(gold_path, "r", encoding="utf-8") as f:
                gold_data = json.load(f)
            summary = evaluate_gold_tests(gold_data, model_name=model_name, use_llm=use_llm)
            st.session_state.gold_summary = summary
            st.success(f"✅ Gold test run complete — Overall Accuracy: {summary['overall_accuracy']}%")
            st.json(summary["per_field_accuracy"])
        except Exception as e:
            st.error(f"Failed to run gold tests: {e}")

    st.markdown("---")
    st.header("📈 Compute & Commit Field Weights")

    min_w = st.number_input("Min weight", value=0.4, step=0.05, key="min_weight")
    max_w = st.number_input("Max weight", value=1.6, step=0.05, key="max_weight")

    if st.button("⚖️ Compute weights", key="compute_weights_btn"):
        try:
            summary = st.session_state.get("gold_summary")
            if not summary:
                st.warning("Run gold tests first!")
            else:
                from training.field_weighting import compute_weights_from_accuracy, sanitize_weights
                weights = compute_weights_from_accuracy(summary["per_field_accuracy"], min_w=min_w, max_w=max_w)
                ok, msg = sanitize_weights(weights, min_w=min_w, max_w=max_w)
                if not ok:
                    st.error(f"Sanitization failed: {msg}")
                else:
                    with open("training/field_weights.json", "w", encoding="utf-8") as f:
                        json.dump(weights, f, indent=2)
                    st.success("✅ Field weights computed and saved!")
        except Exception as e:
            st.error(f"Error computing weights: {e}")

    st.markdown("---")
    st.header("🧩 Learned Rule Suggestions")

    if st.button("💡 Suggest new rules from Gold Tests"):
        try:
            with open(gold_path, "r", encoding="utf-8") as f:
                gold_data = json.load(f)
            candidates = suggest_rules_from_gold(gold_data, model_name=model_name, use_llm=use_llm)
            save_rule_suggestions(candidates)
            st.success(f"Suggested {len(candidates)} candidate rules. Saved to training/rule_candidates.json")
        except Exception as e:
            st.error(f"Failed to suggest rules: {e}")

    if st.button("🧹 Sanitize Suggested Rules"):
        try:
            with open("training/rule_candidates.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            ok, msg, cleaned = sanitize_rules(data.get("rules", []))
            st.info(msg)
            if ok:
                st.session_state.sanitized_rules = cleaned
        except Exception as e:
            st.error(f"Sanitization error: {e}")

    st.markdown("---")
    st.header("☁️ Commit to GitHub")

    default_branch_name = f"bactai-weights-update-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    new_branch_name = st.text_input("New branch name (auto timestamped)", value=default_branch_name, key="branch_name")

    if st.button("⬆️ Commit Field Weights to GitHub"):
        try:
            commit_to_github(
                repo="EphraimAsad/llmrag",
                branch_name=new_branch_name,
                file_path="training/field_weights.json",
                commit_msg=f"Update field weights ({datetime.now().strftime('%Y-%m-%d %H:%M')})",
                pr_title="Auto: Update Field Weights"
            )
            st.success("✅ Field weights committed successfully.")
        except Exception as e:
            st.error(f"Commit failed: {e}")

    if st.button("⬆️ Commit Learned Rules to GitHub"):
        try:
            commit_to_github(
                repo="EphraimAsad/llmrag",
                branch_name=f"bactai-learned-rules-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                file_path="training/learned_rules.json",
                commit_msg=f"Add/update learned rules ({datetime.now().strftime('%Y-%m-%d %H:%M')})",
                pr_title="Auto: Add Learned Rules"
            )
            st.success("✅ Learned rules committed successfully.")
        except Exception as e:
            st.error(f"Commit failed: {e}")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; font-size:14px;'>Created by <b>Zain</b> | www.linkedin.com/in/zain-asad-1998EPH</div>", unsafe_allow_html=True)
