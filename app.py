import streamlit as st
import pandas as pd
import os
import json
from datetime import datetime
from fpdf import FPDF

# Engine imports
from engine import BacteriaIdentifier
from engine.parser_llm import evaluate_gold_tests
from engine.parser_rules_runtime import apply_rules
from training.rules_suggester import suggest_rules_from_gold, save_rule_suggestions
from training.rules_sanitizer import sanitize_rules

# --- CONFIG ---
st.set_page_config(page_title="BactAI-D Assistant", layout="wide")

# --- SIDEBAR SETUP ---
st.sidebar.title("🧫 BactAI-D Assistant")
page = st.sidebar.radio("Navigation", ["🔍 Identify", "🧠 Training"])

# =========================================================
#                    IDENTIFICATION TAB
# =========================================================
if page == "🔍 Identify":
    # --- LOAD DATABASE ---
    @st.cache_data
    def load_data(path):
        df = pd.read_excel(path)
        df.columns = [c.strip() for c in df.columns]
        return df

    primary_path = os.path.join("data", "bacteria_db.xlsx")
    fallback_path = "bacteria_db.xlsx"
    data_path = primary_path if os.path.exists(primary_path) else fallback_path

    if not os.path.exists(data_path):
        st.error("❌ Database not found.")
        st.stop()

    db = load_data(data_path)
    eng = BacteriaIdentifier(db)

    st.title("🔍 BactAI-D Identification")
    st.markdown("Input your test results using the sidebar and click *Identify*.")

    # --- Sidebar input setup ---
    test_fields = [c for c in db.columns if c != "Genus"]
    user_input = {}

    for f in test_fields:
        if f in ["Shape", "Colony Morphology", "Media Grown On"]:
            user_input[f] = st.sidebar.text_input(f, "")
        elif f == "Growth Temperature":
            user_input[f] = st.sidebar.text_input(f + " (°C)", "")
        else:
            user_input[f] = st.sidebar.selectbox(f, ["Unknown", "Positive", "Negative", "Variable"], index=0)

    if st.sidebar.button("🔎 Identify"):
        with st.spinner("Analyzing results..."):
            results = eng.identify(user_input)
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
                    columns=[
                        "Genus",
                        "Confidence",
                        "True Confidence (All Tests)",
                        "Reasoning",
                        "Next Tests",
                        "Extra Notes"
                    ],
                )
                st.session_state["results"] = results_df

    # --- Display results ---
    if "results" in st.session_state and not st.session_state["results"].empty:
        st.info("Percentages are based on tests entered. True confidence considers all possible fields.")
        for _, row in st.session_state["results"].iterrows():
            conf_val = int(row["Confidence"].replace("%", ""))
            color = "🟢" if conf_val >= 75 else "🟡" if conf_val >= 50 else "🔴"
            header = f"**{row['Genus']}** — {color} {row['Confidence']}"
            with st.expander(header):
                st.markdown(f"**Reasoning:** {row['Reasoning']}")
                st.markdown(f"**Next Tests:** {row['Next Tests']}")
                st.markdown(f"**True Confidence:** {row['True Confidence (All Tests)']}")
                if row["Extra Notes"]:
                    st.markdown(f"**Notes:** {row['Extra Notes']}")

    # --- PDF EXPORT ---
    def export_pdf(results_df, user_input):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "BactAI-D Identification Report", ln=True, align="C")
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Entered Test Results:", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for k, v in user_input.items():
            pdf.multi_cell(0, 6, f"- {k}: {v}")
        pdf.ln(6)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Top Possible Matches:", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for _, row in results_df.iterrows():
            pdf.multi_cell(0, 7, f"- {row['Genus']} ({row['Confidence']})")
            pdf.multi_cell(0, 6, f"  Reasoning: {row['Reasoning']}")
        out = "BactAI_Report.pdf"
        pdf.output(out)
        return out

    if "results" in st.session_state and not st.session_state["results"].empty:
        if st.button("📄 Export PDF"):
            pdf_path = export_pdf(st.session_state["results"], user_input)
            with open(pdf_path, "rb") as f:
                st.download_button("⬇️ Download PDF", f, file_name="BactAI_Report.pdf")

# =========================================================
#                       TRAINING TAB
# =========================================================
elif page == "🧠 Training":
    st.title("🧠 BactAI-D Training Suite")
    st.caption("Run gold-standard evaluations, compute weights, and learn new parsing rules.")

    st.divider()
    st.subheader("📘 Gold Test Evaluation")

    gold_path = st.text_input("Gold tests path", value="training/gold_tests.json", key="gold_path_input")
    use_llm = st.toggle("Use LLM fallback", value=True, key="toggle_llm_train")
    model_name = st.text_input("Ollama model", value=os.getenv("OLLAMA_MODEL", "deepseek-coder:6.7b"), key="model_input")

    if st.button("▶️ Run Gold Tests", key="run_gold_tests_btn"):
        try:
            with open(gold_path, "r", encoding="utf-8") as f:
                gold_data = json.load(f)
            summary = evaluate_gold_tests(gold_data, model_name=model_name, use_llm=use_llm)
            st.session_state["gold_summary"] = summary
            st.success(f"✅ Overall accuracy: {summary['overall_accuracy']}%")
            st.json(summary["per_field_accuracy"])
        except Exception as e:
            st.error(f"Error running gold tests: {e}")

    # =====================================================
    #  RULE SUGGESTION + SANITIZATION SECTION
    # =====================================================
    st.divider()
    st.subheader("🧩 Rule Learning")

    if st.button("💡 Suggest new rules", key="suggest_rules_btn"):
        try:
            with open(gold_path, "r", encoding="utf-8") as f:
                gold_data = json.load(f)
            model = st.session_state.get("model_input", "deepseek-coder:6.7b")
            rules = suggest_rules_from_gold(gold_data, model, use_llm=use_llm)
            save_rule_suggestions(rules)
            st.success(f"✅ Suggested {len(rules)} new rule candidates.")
        except Exception as e:
            st.error(f"Rule suggestion failed: {e}")

    if st.button("🧹 Sanitize rule candidates", key="sanitize_rules_btn"):
        try:
            with open("training/rule_candidates.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            ok, msg, cleaned = sanitize_rules(data.get("rules", []))
            st.info(msg)
            if ok:
                st.session_state["sanitized_rules"] = cleaned
                st.success(f"{len(cleaned)} sanitized rules ready for merging.")
        except Exception as e:
            st.error(f"Sanitization failed: {e}")

    # =====================================================
    #  GITHUB COMMIT SECTION (for weights or rules)
    # =====================================================
    st.divider()
    st.subheader("📦 Commit to GitHub")

    repo = st.text_input("GitHub repository (username/repo)", key="gh_repo_inp")
    token = st.text_input("GitHub token", type="password", key="gh_token_inp")

    # Auto-timestamped branch
    default_branch_name = f"bactai-update-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    new_branch_name = st.text_input("New branch name (auto-generated)", value=default_branch_name, key="gh_branch_inp")

    commit_type = st.selectbox("What to commit?", ["field_weights.json", "learned_rules.json"], key="gh_commit_type")

    if st.button("⬆️ Commit to GitHub", key="gh_commit_btn"):
        try:
            import requests

            headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}
            api_base = f"https://api.github.com/repos/{repo}"

            file_path = f"training/{commit_type}"
            if not os.path.exists(file_path):
                st.error(f"{file_path} not found.")
                st.stop()

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Get default branch (usually main)
            repo_info = requests.get(api_base, headers=headers).json()
            default_branch = repo_info.get("default_branch", "main")

            # Get latest commit SHA
            ref_url = f"{api_base}/git/ref/heads/{default_branch}"
            ref_data = requests.get(ref_url, headers=headers).json()
            latest_sha = ref_data["object"]["sha"]

            # Create new branch
            create_branch_url = f"{api_base}/git/refs"
            branch_data = {"ref": f"refs/heads/{new_branch_name}", "sha": latest_sha}
            requests.post(create_branch_url, headers=headers, json=branch_data)

            # Create/update file contents
            put_url = f"{api_base}/contents/{file_path}"
            encoded = content.encode("utf-8").decode("utf-8")
            commit_message = f"Auto-update {commit_type} ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})"
            payload = {
                "message": commit_message,
                "content": content.encode("utf-8").decode("utf-8"),
                "branch": new_branch_name
            }
            requests.put(put_url, headers=headers, json=payload)

            # Open pull request
            pr_url = f"{api_base}/pulls"
            pr_payload = {
                "title": f"Update {commit_type} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "head": new_branch_name,
                "base": default_branch,
                "body": f"Auto-commit of {commit_type} from Streamlit Cloud."
            }
            pr_resp = requests.post(pr_url, headers=headers, json=pr_payload)

            if pr_resp.status_code == 201:
                pr_data = pr_resp.json()
                st.success(f"✅ Pull request created: {pr_data['html_url']}")
            else:
                st.warning(f"Open PR failed: {pr_resp.text}")

        except Exception as e:
            st.error(f"GitHub commit failed: {e}")

# =========================================================
#                        FOOTER
# =========================================================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; font-size:14px;'>Created by <b>Zain</b> — BactAI-D Project</div>", unsafe_allow_html=True)
