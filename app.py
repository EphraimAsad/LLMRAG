import streamlit as st
import pandas as pd
import os
import json
from datetime import datetime
from fpdf import FPDF

from engine import BacteriaIdentifier
from engine.parser_llm import evaluate_gold_tests
from training.rules_suggester import suggest_rules_from_gold
from training.rules_sanitizer import sanitize_rules

# --- CONFIG ---
st.set_page_config(page_title="BactAI-D Assistant", layout="wide")

# --- TAB SETUP ---
tabs = st.tabs(["🧫 Identify", "💬 Chat", "🧠 Training"])

# ============================================================
# 🧫 TAB 1 — MAIN IDENTIFICATION INTERFACE
# ============================================================
with tabs[0]:
    # --- LOAD DATABASE ---
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

    st.sidebar.caption(f"📅 Database last updated: {datetime.fromtimestamp(last_modified).strftime('%Y-%m-%d %H:%M:%S')}")

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

    if "reset_trigger" in st.session_state and st.session_state["reset_trigger"]:
        for key in list(st.session_state.user_input.keys()):
            st.session_state.user_input[key] = "Unknown"
        for key in list(st.session_state.keys()):
            if key not in ["user_input", "results", "reset_trigger"]:
                if isinstance(st.session_state[key], list):
                    st.session_state[key] = []
                else:
                    st.session_state[key] = "Unknown"
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
        st.session_state["reset_trigger"] = True
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

    if not st.session_state.results.empty:
        st.info("Percentages based upon options entered. True confidence percentage shown within each expanded result.")
        for _, row in st.session_state.results.iterrows():
            confidence_value = int(row["Confidence"].replace("%", ""))
            confidence_color = "🟢" if confidence_value >= 75 else "🟡" if confidence_value >= 50 else "🔴"
            header = f"**{row['Genus']}** — {confidence_color} {row['Confidence']}"
            with st.expander(header):
                st.markdown(f"**Reasoning:** {row['Reasoning']}")
                st.markdown(f"**Top 3 Next Tests to Differentiate:** {row['Next Tests']}")
                st.markdown(f"**True Confidence (All Tests):** {row['True Confidence (All Tests)']}")
                if row["Extra Notes"]:
                    st.markdown(f"**Notes:** {row['Extra Notes']}")

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

    if not st.session_state.results.empty:
        if st.button("📄 Export Results to PDF"):
            pdf_path = export_pdf(st.session_state.results, st.session_state.user_input)
            with open(pdf_path, "rb") as f:
                st.download_button("⬇️ Download PDF", f, file_name="BactAI-d_Report.pdf")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center; font-size:14px;'>Created by <b>Zain</b></div>", unsafe_allow_html=True)


# ============================================================
# 💬 TAB 2 — CHAT UI
# ============================================================
with tabs[1]:
    st.title("💬 BactAI-D Chat Assistant")
    st.write("Ask any microbiology or app-related questions below:")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("You:", key="chat_input")
    if st.button("Send"):
        if user_input:
            st.session_state.chat_history.append(("user", user_input))
            response = "I'm still learning! But I can help explain microbiological concepts soon."
            st.session_state.chat_history.append(("assistant", response))
    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"**🧍‍♂️ You:** {msg}")
        else:
            st.markdown(f"**🤖 BactAI-D:** {msg}")


# ============================================================
# 🧠 TAB 3 — TRAINING
# ============================================================
with tabs[2]:
    st.title("🧠 BactAI-D Trainer")
    st.markdown("Use this section to run gold tests, compute field weights, and train rule patterns.")

    col1, col2 = st.columns([2, 1])
    with col1:
        gold_path = st.text_input("Gold tests path", value="training/gold_tests.json", key="gold_path_inp")
        use_llm = st.toggle("Use LLM fallback", value=True, key="use_llm_tog")
    with col2:
        model_name = st.text_input("Ollama model", value=os.getenv("OLLAMA_MODEL", "deepseek-coder:6.7b"), key="ollama_model_inp")

    run_btn = st.button("▶️ Run gold tests")
    if run_btn:
        try:
            with open(gold_path, "r", encoding="utf-8") as f:
                gold_data = json.load(f)
            summary = evaluate_gold_tests(gold_data, model_name=model_name, use_llm=use_llm)  # fixed argument here ✅
            st.session_state["gold_summary"] = summary
            st.success(f"✅ Completed — Overall accuracy: {summary['overall_accuracy']}%")
            st.json(summary["per_field_accuracy"])
        except Exception as e:
            st.error(f"Error running gold tests: {e}")

    # Suggest rule candidates
    if st.button("💡 Suggest new rules"):
        try:
            with open(gold_path, "r", encoding="utf-8") as f:
                gold_data = json.load(f)
            suggestions = suggest_rules_from_gold(gold_data, model_name=model_name, use_llm=use_llm)
            with open("training/rule_candidates.json", "w", encoding="utf-8") as f:
                json.dump({"rules": suggestions}, f, indent=2)
            st.success(f"✅ Suggested {len(suggestions)} new rules. Saved to training/rule_candidates.json")
        except Exception as e:
            st.error(f"Error suggesting rules: {e}")

    # Sanitize rules
    if st.button("🧹 Sanitize suggested rules"):
        try:
            with open("training/rule_candidates.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            ok, msg, cleaned = sanitize_rules(data.get("rules", []))
            st.info(msg)
            if ok:
                st.session_state["sanitized_rules"] = cleaned
        except Exception as e:
            st.error(f"Error sanitizing rules: {e}")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center; font-size:14px;'>Created by <b>Zain</b></div>", unsafe_allow_html=True)
