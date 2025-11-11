import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from fpdf import FPDF

from engine import BacteriaIdentifier
from engine.parser_llm import evaluate_gold_tests, smart_parse
from training.rules_suggester import suggest_rules_from_gold, save_rule_suggestions
from training.rules_sanitizer import sanitize_rules
from training.github_sync import commit_to_github  # your GitHub helper function

# --- CONFIG ---
st.set_page_config(page_title="BactAI-D Assistant", layout="wide")

# --- LOAD DATA with auto-reload ---
@st.cache_data
def load_data(path, last_modified):
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    return df

# Resolve path (prefer ./data/bacteria_db.xlsx)
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

# --- SIDEBAR ---
st.sidebar.caption(f"📅 DB updated: {datetime.fromtimestamp(last_modified).strftime('%Y-%m-%d %H:%M:%S')}")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["🧫 Identification", "💬 Chat / Notes", "🧠 Training"])

# =====================================================
# TAB 1 — IDENTIFICATION
# =====================================================
with tab1:
    st.title("🧫 BactAI-D: Intelligent Bacteria Identification Assistant")
    st.markdown("Use the sidebar to input biochemical and morphological results.")

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
            parts = str(v).replace("/", ";").split(";")
            for p in parts:
                clean = p.strip()
                if clean and clean not in vals:
                    vals.append(clean)
        vals.sort()
        return vals

    st.sidebar.markdown(
        "<div style='background-color:#1565C0; padding:12px; border-radius:10px;'>"
        "<h3 style='text-align:center; color:white; margin:0;'>🔬 Input Test Results</h3>"
        "</div>",
        unsafe_allow_html=True
    )

    with st.sidebar.expander("🧫 Morphological Tests", expanded=True):
        for field in MORPH_FIELDS:
            if field in ["Shape", "Colony Morphology", "Media Grown On"]:
                options = get_unique_values(field)
                selected = st.multiselect(field, options, default=[], key=field)
                st.session_state.user_input[field] = "; ".join(selected) if selected else "Unknown"
            else:
                st.session_state.user_input[field] = st.selectbox(field, ["Unknown", "Positive", "Negative", "Variable"], key=field)

    with st.sidebar.expander("🧪 Enzyme Tests", expanded=False):
        for field in ENZYME_FIELDS:
            st.session_state.user_input[field] = st.selectbox(field, ["Unknown", "Positive", "Negative", "Variable"], key=field)

    with st.sidebar.expander("🍬 Carbohydrate Fermentation Tests", expanded=False):
        for field in SUGAR_FIELDS:
            st.session_state.user_input[field] = st.selectbox(field, ["Unknown", "Positive", "Negative", "Variable"], key=field)

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
                st.session_state.user_input[field] = st.selectbox(field, ["Unknown"] + options, key=field)
            elif field == "Growth Temperature":
                st.session_state.user_input[field] = st.text_input(field + " (°C)", "", key=field)
            else:
                st.session_state.user_input[field] = st.selectbox(field, ["Unknown", "Positive", "Negative", "Variable"], key=field)

    if st.sidebar.button("🔄 Reset All Inputs"):
        for key in list(st.session_state.user_input.keys()):
            st.session_state.user_input[key] = "Unknown"
        st.session_state.results = pd.DataFrame()
        st.experimental_rerun()

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
        st.info("Percentages based on entered tests. True confidence uses all available tests.")
        for _, row in st.session_state.results.iterrows():
            conf_val = int(row["Confidence"].replace("%", ""))
            color = "🟢" if conf_val >= 75 else "🟡" if conf_val >= 50 else "🔴"
            header = f"**{row['Genus']}** — {color} {row['Confidence']}"
            with st.expander(header):
                st.markdown(f"**Reasoning:** {row['Reasoning']}")
                st.markdown(f"**Next Tests:** {row['Next Tests']}")
                st.markdown(f"**True Confidence:** {row['True Confidence (All Tests)']}")

    # PDF Export
    def export_pdf(results_df, user_input):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "BactAI-D Identification Report", ln=True, align="C")
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.ln(6)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Entered Results:", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for k, v in user_input.items():
            pdf.multi_cell(0, 6, f"- {k}: {v}")
        pdf.ln(6)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Top Matches:", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for _, row in results_df.iterrows():
            pdf.multi_cell(0, 7, f"- {row['Genus']} — {row['Confidence']} (True: {row['True Confidence (All Tests)']})")
        pdf.output("BactAI-D_Report.pdf")
        return "BactAI-D_Report.pdf"

    if not st.session_state.results.empty:
        if st.button("📄 Export Results to PDF"):
            pdf_path = export_pdf(st.session_state.results, st.session_state.user_input)
            with open(pdf_path, "rb") as f:
                st.download_button("⬇️ Download PDF", f, file_name="BactAI-D_Report.pdf")

# =====================================================
# TAB 2 — CHAT / NOTES
# =====================================================
with tab2:
    st.title("💬 BactAI-D Chat / Notes")
    st.markdown("Use this space for personal notes, summaries, or conversational brainstorming.")
    user_text = st.text_area("🧠 Write or brainstorm below:", height=250)
    if st.button("💾 Save Notes"):
        os.makedirs("training", exist_ok=True)
        path = f"training/notes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(user_text)
        st.success(f"Saved your notes to `{path}`")

# =====================================================
# TAB 3 — TRAINING / GOLD TESTS
# =====================================================
with tab3:
    st.title("🧠 Model Training & Gold Standard Evaluation")

    col1, col2 = st.columns([2, 1])
    with col1:
        gold_path = st.text_input("📂 Gold Tests Path", value="training/gold_tests.json", key="gold_path_input")
        use_llm = st.toggle("Use LLM Fallback", value=True, key="use_llm_toggle")
    with col2:
        model_name = st.text_input("🧠 Ollama Model", value=os.getenv("OLLAMA_MODEL", "deepseek-coder:6.7b"), key="model_name_input")

    run_btn = st.button("▶️ Run Gold Tests")
    if run_btn:
        if not os.path.exists(gold_path):
            st.error(f"Gold file not found: {gold_path}")
        else:
            with open(gold_path, "r", encoding="utf-8") as f:
                gold_data = json.load(f)
            with st.spinner("Evaluating model against gold tests..."):
                summary = evaluate_gold_tests(gold_data, model_name=model_name, use_llm=use_llm)
                st.session_state["gold_summary"] = summary
                st.success(f"✅ Overall Accuracy: {summary['overall_accuracy']}%")

    # --- Suggest new rules ---
    if "gold_summary" in st.session_state:
        st.divider()
        st.markdown("### 🔍 Suggest New Parsing Rules")
        if st.button("✨ Generate Rule Suggestions"):
            with open(gold_path, "r", encoding="utf-8") as f:
                gold_data = json.load(f)
            with st.spinner("Analyzing failed cases and generating rule candidates..."):
                suggestions = suggest_rules_from_gold(gold_data, model_name=model_name, use_llm=use_llm)
                save_rule_suggestions(suggestions)
                st.success(f"Generated {len(suggestions)} new rule candidates.")
                st.session_state["rule_suggestions"] = suggestions

    # --- Sanitize suggested rules ---
    if "rule_suggestions" in st.session_state:
        st.divider()
        st.markdown("### 🧹 Sanitize Rule Candidates")
        if st.button("🧼 Validate & Sanitize Rules"):
            ok, msg, cleaned = sanitize_rules(st.session_state["rule_suggestions"])
            st.info(msg)
            if ok:
                st.session_state["sanitized_rules"] = cleaned
                st.success(f"Validated {len(cleaned)} rules successfully.")

    # --- Commit to GitHub ---
    if "sanitized_rules" in st.session_state:
        st.divider()
        st.markdown("### 🔗 Commit Sanitized Rules to GitHub")
        default_branch_name = f"bactai-learned-rules-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        repo_name = st.text_input("GitHub Repo (owner/repo)", value=os.getenv("GITHUB_REPO", "EphraimAsad/BactAI-D"))
        new_branch_name = st.text_input("New Branch Name", value=default_branch_name)
        commit_message = st.text_input("Commit Message", value="feat: add/update learned rules")

        if st.button("🚀 Commit to GitHub"):
            data = {"version": 1, "rules": st.session_state["sanitized_rules"]}
            os.makedirs("training", exist_ok=True)
            path = "training/learned_rules.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            with st.spinner("Pushing to GitHub..."):
                ok, msg = commit_to_github(
                    repo=repo_name,
                    branch=new_branch_name,
                    file_path=path,
                    commit_message=commit_message,
                    pr_title="feat: add/update learned rules",
                )
                if ok:
                    st.success("✅ Pushed and PR created successfully!")
                else:
                    st.error(f"❌ {msg}")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; font-size:14px;'>Created by <b>Zain</b></div>", unsafe_allow_html=True)
