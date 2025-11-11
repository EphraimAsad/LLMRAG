import streamlit as st
import pandas as pd
import re
import os
from fpdf import FPDF
from datetime import datetime
from engine import BacteriaIdentifier
from engine.parser_llm import smart_parse
from engine.parser_rules import parse_and_canonicalize as parse_rules_only

# --- CONFIG ---
st.set_page_config(page_title="BactAI-D Assistant", layout="wide")

# --- LOAD DATA ---
@st.cache_data
def load_data(path, last_modified):
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    return df

# Resolve path
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
st.sidebar.caption(f"📅 Database last updated: {datetime.fromtimestamp(last_modified).strftime('%Y-%m-%d %H:%M:%S')}")

# --- PAGE HEADER ---
st.title("🧫 BactAI-D: Intelligent Bacteria Identification Assistant")
st.markdown("Choose an input mode: **Form**, **Text Parser**, or **Training** for gold tests.")

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

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

def export_pdf(results_df, user_input):
    def safe_text(text):
        text = str(text).replace("•", "-").replace("—", "-").replace("–", "-")
        return text.encode("latin-1", "replace").decode("latin-1")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "BactAI-d Identification Report", ln=True, align="C")

    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Entered Test Results:", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for k, v in user_input.items():
        pdf.multi_cell(0, 6, safe_text(f"- {k}: {v}"))

    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Top Possible Matches:", ln=True)
    pdf.set_font("Helvetica", "", 10)

    for _, row in results_df.iterrows():
        pdf.multi_cell(0, 7, safe_text(f"- {row['Genus']} — Confidence: {row['Confidence']} (True: {row['True Confidence (All Tests)']})"))
        pdf.multi_cell(0, 6, safe_text(f"  Reasoning: {row['Reasoning']}"))
        if row['Next Tests']:
            pdf.multi_cell(0, 6, safe_text(f"  Next Tests: {row['Next Tests']}"))
        if row['Extra Notes']:
            pdf.multi_cell(0, 6, safe_text(f"  Notes: {row['Extra Notes']}"))
        pdf.ln(3)

    pdf.output("BactAI-d_Report.pdf")
    return "BactAI-d_Report.pdf"

def coerce_growth_temp_for_engine(user_input: dict) -> dict:
    ui = dict(user_input)
    val = str(ui.get("Growth Temperature", "")).strip()
    if "//" in val:
        try:
            low_s, high_s = val.split("//", 1)
            low, high = float(low_s), float(high_s)
            mid = (low + high) / 2.0
            ui["Growth Temperature"] = str(int(mid))
        except Exception:
            pass
    return ui

def run_identification(user_input: dict):
    ui_for_engine = coerce_growth_temp_for_engine(user_input)
    results = eng.identify(ui_for_engine)
    if not results:
        return pd.DataFrame()

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
    return df

# -----------------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------------

tab_form, tab_text, tab_train = st.tabs(["📝 Form Mode", "🧠 Text Mode (Rules + LLM)", "🎓 Training"])

# ----------------------------- FORM MODE -------------------------------------
with tab_form:
    st.markdown("Use the sidebar to input test results.")

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

    if "reset_trigger" in st.session_state and st.session_state["reset_trigger"]:
        for key in list(st.session_state.user_input.keys()):
            st.session_state.user_input[key] = "Unknown"
        st.session_state["reset_trigger"] = False
        st.rerun()

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
                selected = st.multiselect(field, options, default=[], key=f"form_{field}")
                st.session_state.user_input[field] = "; ".join(selected) if selected else "Unknown"
            else:
                st.session_state.user_input[field] = st.selectbox(field, ["Unknown", "Positive", "Negative", "Variable"], index=0, key=f"form_{field}")

    with st.sidebar.expander("🧪 Enzyme Tests", expanded=False):
        for field in ENZYME_FIELDS:
            st.session_state.user_input[field] = st.selectbox(field, ["Unknown", "Positive", "Negative", "Variable"], index=0, key=f"form_{field}")

    with st.sidebar.expander("🍬 Carbohydrate Fermentation Tests", expanded=False):
        for field in SUGAR_FIELDS:
            st.session_state.user_input[field] = st.selectbox(field, ["Unknown", "Positive", "Negative", "Variable"], index=0, key=f"form_{field}")

    with st.sidebar.expander("🧬 Other Tests", expanded=False):
        for field in db.columns:
            if field in ["Genus"] + MORPH_FIELDS + ENZYME_FIELDS + SUGAR_FIELDS:
                continue
            if field == "Haemolysis Type":
                options = get_unique_values(field)
                selected = st.multiselect(field, options, default=[], key=f"form_{field}")
                st.session_state.user_input[field] = "; ".join(selected) if selected else "Unknown"
            elif field == "Oxygen Requirement":
                options = get_unique_values(field)
                st.session_state.user_input[field] = st.selectbox(field, ["Unknown"] + options, index=0, key=f"form_{field}")
            elif field == "Growth Temperature":
                st.session_state.user_input[field] = st.text_input(field + " (°C)", "", key=f"form_{field}")
            else:
                st.session_state.user_input[field] = st.selectbox(field, ["Unknown", "Positive", "Negative", "Variable"], index=0, key=f"form_{field}")

    if st.sidebar.button("🔄 Reset All Inputs"):
        st.session_state["reset_trigger"] = True
        st.rerun()

    if st.sidebar.button("🔍 Identify"):
        with st.spinner("Analyzing results..."):
            results_df = run_identification(st.session_state.user_input)
            if results_df.empty:
                st.error("No matches found.")
            else:
                st.session_state.results = results_df

    if not st.session_state.results.empty:
        st.info("Percentages based upon options entered.")
        for _, row in st.session_state.results.iterrows():
            confidence_value = int(row["Confidence"].replace("%", ""))
            confidence_color = "🟢" if confidence_value >= 75 else "🟡" if confidence_value >= 50 else "🔴"
            header = f"**{row['Genus']}** — {confidence_color} {row['Confidence']}"
            with st.expander(header):
                st.markdown(f"**Reasoning:** {row['Reasoning']}")
                st.markdown(f"**Next Tests:** {row['Next Tests']}")
                st.markdown(f"**True Confidence:** {row['True Confidence (All Tests)']}")

    if not st.session_state.results.empty:
        if st.button("📄 Export Results to PDF"):
            pdf_path = export_pdf(st.session_state.results, st.session_state.user_input)
            with open(pdf_path, "rb") as f:
                st.download_button("⬇️ Download PDF", f, file_name="BactAI-d_Report.pdf")

# ----------------------------- TEXT MODE -------------------------------------
with tab_text:
    st.markdown("Paste microbiology description below.")
    colA, colB = st.columns([2,1])
    with colA:
        text_input = st.text_area("Free-text description", height=200)
    with colB:
        use_llm = st.toggle("Use LLM fallback", value=True)
        model_name = st.text_input("Ollama model", value=os.getenv("OLLAMA_MODEL", "deepseek-r1:latest"))

    parse_btn = st.button("🧠 Parse Text")
    identify_btn = st.button("🔍 Identify from Parsed")

    if "parsed_record" not in st.session_state:
        st.session_state.parsed_record = None

    if parse_btn and text_input.strip():
        with st.spinner("Parsing text..."):
            st.session_state.parsed_record = smart_parse(text_input, model=model_name) if use_llm else parse_rules_only(text_input)

    if st.session_state.parsed_record:
        st.subheader("Parsed Record")
        st.json(st.session_state.parsed_record)

    if identify_btn and st.session_state.parsed_record:
        with st.spinner("Analyzing parsed results..."):
            results_df = run_identification(st.session_state.parsed_record)
            if results_df.empty:
                st.error("No matches found.")
            else:
                for _, row in results_df.iterrows():
                    header = f"**{row['Genus']}** — {row['Confidence']}"
                    with st.expander(header):
                        st.markdown(f"**Reasoning:** {row['Reasoning']}")

# ----------------------------- TRAINING TAB ----------------------------------
with tab_train:
    from training.gold_eval import run_gold_tests
    st.markdown("Run your **Gold Tests** to measure parser accuracy and train weights.")

    col1, col2 = st.columns([2,1])
    with col1:
        gold_path = st.text_input("Gold tests path", value="training/gold_tests.json")
        use_llm = st.toggle("Use LLM fallback", value=True)
    with col2:
        model_name = st.text_input("Ollama model", value=os.getenv("OLLAMA_MODEL", "deepseek-r1:latest"))
    run_btn = st.button("▶️ Run gold tests")

    if run_btn:
        try:
            with st.spinner("Evaluating gold tests..."):
                from training.gold_eval import run_gold_tests
                summary, df_cases, df_fields = run_gold_tests(gold_path, use_llm=use_llm, model=model_name)
            st.subheader(f"Overall accuracy: **{summary['overall_accuracy_percent']}%** ({summary['cases_count']} cases)")
            st.markdown("### Per-field accuracy:")
            st.dataframe(df_fields, use_container_width=True)
            st.markdown("### Per-case results:")
            st.dataframe(df_cases, use_container_width=True)
            st.download_button("⬇️ Download per-field CSV", df_fields.to_csv(index=False), file_name="per_field_accuracy.csv")
            st.download_button("⬇️ Download per-case CSV", df_cases.to_csv(index=False), file_name="per_case_accuracy.csv")
        except Exception as e:
            st.error(f"Error: {e}")

# --- FOOTER ---
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; font-size:14px;'>Created by <b>Zain</b></div>", unsafe_allow_html=True)
