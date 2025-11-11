import streamlit as st
import pandas as pd
import re
import os
import json
from datetime import datetime
from fpdf import FPDF

# Engine imports
from engine import BacteriaIdentifier
from engine.parser_llm import evaluate_gold_tests, smart_parse
from training.rules_suggester import suggest_rules_from_gold, save_rule_suggestions
from training.rules_sanitizer import sanitize_rules

# ---------------------------
# STREAMLIT CONFIG
# ---------------------------
st.set_page_config(page_title="🧫 BactAI-D Assistant", layout="wide")

# ---------------------------
# LOAD DATABASE
# ---------------------------
@st.cache_data
def load_data(path, last_modified):
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    return df

primary_path = os.path.join("data", "bacteria_db.xlsx")
fallback_path = "bacteria_db.xlsx"
data_path = primary_path if os.path.exists(primary_path) else fallback_path

try:
    last_modified = os.path.getmtime(data_path)
except FileNotFoundError:
    st.error("Database file not found.")
    st.stop()

db = load_data(data_path, last_modified)
eng = BacteriaIdentifier(db)

# Sidebar meta
st.sidebar.caption(f"📅 DB updated: {datetime.fromtimestamp(last_modified).strftime('%Y-%m-%d %H:%M:%S')}")

# ---------------------------
# APP TABS
# ---------------------------
tab1, tab2 = st.tabs(["💬 Identify / Chat", "🧠 Training"])

# ============================================================
# TAB 1 – Chat / Identify
# ============================================================
with tab1:
    st.title("🧫 BactAI-D: Intelligent Bacteria Identification Assistant")
    st.markdown("Use the sidebar to input your biochemical and morphological results.")

    MORPH_FIELDS = ["Gram Stain","Shape","Colony Morphology","Media Grown On","Motility","Capsule","Spore Formation"]
    ENZYME_FIELDS = ["Catalase","Oxidase","Coagulase","Lipase Test"]
    SUGAR_FIELDS = [
        "Glucose Fermentation","Lactose Fermentation","Sucrose Fermentation","Maltose Fermentation",
        "Mannitol Fermentation","Sorbitol Fermentation","Xylose Fermentation","Rhamnose Fermentation",
        "Arabinose Fermentation","Raffinose Fermentation","Trehalose Fermentation","Inositol Fermentation"
    ]

    if "user_input" not in st.session_state:
        st.session_state.user_input = {}
    if "results" not in st.session_state:
        st.session_state.results = pd.DataFrame()

    # Reset
    if st.sidebar.button("🔄 Reset All Inputs"):
        st.session_state.user_input = {}
        st.session_state.results = pd.DataFrame()
        st.rerun()

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

    # Sidebar inputs
    st.sidebar.markdown("### 🧫 Morphological Tests")
    for field in MORPH_FIELDS:
        if field in ["Shape","Colony Morphology","Media Grown On"]:
            opts = get_unique_values(field)
            sel = st.sidebar.multiselect(field, opts)
            st.session_state.user_input[field] = "; ".join(sel) if sel else "Unknown"
        else:
            st.session_state.user_input[field] = st.sidebar.selectbox(field, ["Unknown","Positive","Negative","Variable"])

    st.sidebar.markdown("### 🧪 Enzyme Tests")
    for field in ENZYME_FIELDS:
        st.session_state.user_input[field] = st.sidebar.selectbox(field, ["Unknown","Positive","Negative","Variable"])

    st.sidebar.markdown("### 🍬 Sugar Fermentation Tests")
    for field in SUGAR_FIELDS:
        st.session_state.user_input[field] = st.sidebar.selectbox(field, ["Unknown","Positive","Negative","Variable"])

    st.sidebar.markdown("### 🧬 Other Tests")
    for field in db.columns:
        if field in ["Genus"] + MORPH_FIELDS + ENZYME_FIELDS + SUGAR_FIELDS:
            continue
        if field == "Haemolysis Type":
            opts = get_unique_values(field)
            sel = st.sidebar.multiselect(field, opts)
            st.session_state.user_input[field] = "; ".join(sel) if sel else "Unknown"
        elif field == "Oxygen Requirement":
            opts = get_unique_values(field)
            st.session_state.user_input[field] = st.sidebar.selectbox(field, ["Unknown"] + opts)
        elif field == "Growth Temperature":
            st.session_state.user_input[field] = st.sidebar.text_input(field + " (°C)")
        else:
            st.session_state.user_input[field] = st.sidebar.selectbox(field, ["Unknown","Positive","Negative","Variable"])

    # Identify
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
                        r.reasoning_factors.get("next_tests",""),
                        r.extra_notes
                    ])
                df = pd.DataFrame(rows, columns=["Genus","Confidence","True Confidence (All Tests)","Reasoning","Next Tests","Extra Notes"])
                st.session_state.results = df

    if not st.session_state.results.empty:
        st.info("Percentages are based on entered tests.")
        for _, row in st.session_state.results.iterrows():
            conf = int(row["Confidence"].replace("%",""))
            col = "🟢" if conf>=75 else "🟡" if conf>=50 else "🔴"
            hdr = f"**{row['Genus']}** — {col} {row['Confidence']}"
            with st.expander(hdr):
                st.markdown(f"**Reasoning:** {row['Reasoning']}")
                st.markdown(f"**Next Tests:** {row['Next Tests']}")
                st.markdown(f"**True Confidence:** {row['True Confidence (All Tests)']}")
                if row["Extra Notes"]:
                    st.markdown(f"**Notes:** {row['Extra Notes']}")

    # Export
    def export_pdf(df, user_input):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica","B",16)
        pdf.cell(0,10,"BactAI-D Identification Report",ln=True,align="C")
        pdf.set_font("Helvetica","",11)
        pdf.cell(0,8,f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",ln=True)
        pdf.ln(4)
        pdf.set_font("Helvetica","B",12)
        pdf.cell(0,8,"Entered Test Results:",ln=True)
        pdf.set_font("Helvetica","",10)
        for k,v in user_input.items():
            pdf.multi_cell(0,6,f"- {k}: {v}")
        pdf.ln(6)
        pdf.set_font("Helvetica","B",12)
        pdf.cell(0,8,"Top Possible Matches:",ln=True)
        pdf.set_font("Helvetica","",10)
        for _,row in df.iterrows():
            pdf.multi_cell(0,7,f"- {row['Genus']} — Confidence: {row['Confidence']}")
            pdf.multi_cell(0,6,f"  Reasoning: {row['Reasoning']}")
            if row['Next Tests']:
                pdf.multi_cell(0,6,f"  Next Tests: {row['Next Tests']}")
            pdf.ln(3)
        path="BactAI_Report.pdf"
        pdf.output(path)
        return path

    if not st.session_state.results.empty:
        if st.button("📄 Export to PDF"):
            pdf_path = export_pdf(st.session_state.results, st.session_state.user_input)
            with open(pdf_path,"rb") as f:
                st.download_button("⬇️ Download PDF",f,file_name="BactAI_Report.pdf")

# ============================================================
# TAB 2 – TRAINING / GOLD TESTS / RULE LEARNING
# ============================================================
with tab2:
    st.title("🧠 BactAI-D Training & Self-Improvement")

    gold_path = st.text_input("Gold tests path", value="training/gold_tests.json", key="gold_path_inp")
    use_llm = st.toggle("Use LLM fallback", value=True, key="use_llm_toggle")
    model_name = st.text_input("Ollama model", value=os.getenv("OLLAMA_MODEL","deepseek-coder:6.7b"), key="ollama_model_inp")

    if st.button("▶️ Run Gold Tests", key="run_gold_tests_btn"):
        with open(gold_path,"r",encoding="utf-8") as f:
            gold_data = json.load(f)
        with st.spinner("Running evaluation..."):
            summary = evaluate_gold_tests(gold_data, model_name=model_name, use_llm=use_llm)
        st.session_state["gold_summary"] = summary
        st.success(f"✅ Overall accuracy: {summary['overall_accuracy']}%")

    # Compute weights
    if "gold_summary" in st.session_state:
        summary = st.session_state["gold_summary"]
        st.subheader("Per-field Accuracy")
        st.json(summary["per_field_accuracy"])

    # Suggest rules
    st.markdown("---")
    st.subheader("📘 Suggest & Sanitize Rules")

    if st.button("✨ Suggest New Rules", key="suggest_rules_btn"):
        with open(gold_path,"r",encoding="utf-8") as f:
            gold_data = json.load(f)
        with st.spinner("Analyzing failed fields..."):
            candidates = suggest_rules_from_gold(gold_data, model_name=model_name, use_llm=use_llm)
        save_rule_suggestions(candidates)
        st.success(f"Suggested {len(candidates)} candidate rules. Saved to training/rule_candidates.json")

    if st.button("🧹 Sanitize Suggested Rules", key="sanitize_rules_btn"):
        if not os.path.exists("training/rule_candidates.json"):
            st.error("No rule_candidates.json found. Run 'Suggest New Rules' first.")
        else:
            with open("training/rule_candidates.json","r",encoding="utf-8") as f:
                data = json.load(f)
            ok, msg, cleaned = sanitize_rules(data.get("rules", []))
            st.info(msg)
            if ok:
                st.session_state["sanitized_rules"] = cleaned
                st.download_button(
                    "⬇️ Download Sanitized Rules JSON",
                    json.dumps({"rules": cleaned}, indent=2),
                    file_name="sanitized_rules.json"
                )

    # Future: Compute weights, commit rules/weights to GitHub, etc.
    st.markdown("---")
    st.caption("Commit to GitHub and advanced learning logic will be added here later.")
