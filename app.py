import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from fpdf import FPDF

# --- IMPORTS ---
from engine import BacteriaIdentifier
from engine.parser_llm import evaluate_gold_tests
from engine.parser_llm import smart_parse

# Training tools
from training.rules_suggester import suggest_rules_from_gold, save_rule_suggestions
from training.rules_sanitizer import sanitize_rules

# GitHub integration
from github import Github
import tempfile


# --------------------------------------------
# STREAMLIT CONFIG
# --------------------------------------------
st.set_page_config(page_title="BactAI-D", layout="wide")
st.sidebar.title("🧫 BactAI-D")

# --------------------------------------------
# LOAD DATABASE
# --------------------------------------------
@st.cache_data
def load_data(path):
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    return df

db_path = "data/bacteria_db.xlsx" if os.path.exists("data/bacteria_db.xlsx") else "bacteria_db.xlsx"
if not os.path.exists(db_path):
    st.error("Database file not found.")
    st.stop()

db = load_data(db_path)
eng = BacteriaIdentifier(db)

# --------------------------------------------
# APP TABS
# --------------------------------------------
tab1, tab2 = st.tabs(["🔍 Identify", "🧠 Training"])

# ===============================================================
# TAB 1 — IDENTIFICATION
# ===============================================================
with tab1:
    st.header("🔍 Identify Unknown Isolate")

    # Collect test results
    st.subheader("Enter Test Results")
    user_input = {}
    for field in db.columns:
        if field == "Genus":
            continue
        user_input[field] = st.text_input(field, "Unknown")

    if st.button("🔬 Identify"):
        results = eng.identify(user_input)
        if not results:
            st.error("No matches found.")
        else:
            df_res = pd.DataFrame([
                [r.genus, f"{r.confidence_percent()}%", f"{r.true_confidence()}%", r.reasoning_paragraph(results)]
                for r in results
            ], columns=["Genus", "Confidence", "True Confidence (All Tests)", "Reasoning"])
            st.dataframe(df_res)


# ===============================================================
# TAB 2 — TRAINING
# ===============================================================
with tab2:
    st.header("🧠 Model Training & Rule Learning")

    st.markdown("""
    This section lets you run gold standard tests, generate field weights, and propose new parsing rules.
    """)

    # --------------------------
    # Gold Test Runner
    # --------------------------
    st.subheader("Run Gold Tests")

    gold_path = st.text_input("Gold tests path", value="training/gold_tests.json")
    use_llm = st.toggle("Use LLM fallback", value=True, key="use_llm_toggle")
    model_name = st.text_input("Ollama model", value=os.getenv("OLLAMA_MODEL", "deepseek-coder:6.7b"))
    run_btn = st.button("▶️ Run Gold Tests")

    if run_btn:
        if not os.path.exists(gold_path):
            st.error("Gold test file not found!")
        else:
            with open(gold_path, "r", encoding="utf-8") as f:
                gold_data = json.load(f)
            st.info("Running gold tests... please wait ⏳")
            summary = evaluate_gold_tests(gold_data, model_name=model_name, use_llm=use_llm)
            st.session_state["gold_summary"] = summary

            st.success(f"✅ Overall accuracy: {summary['overall_accuracy']}%")
            st.write(pd.DataFrame.from_dict(summary["per_field_accuracy"], orient="index", columns=["Accuracy %"]))
            st.write(f"Failed cases: {len(summary['failed_cases'])}")

            with st.expander("View Failed Cases"):
                st.dataframe(pd.DataFrame(summary["failed_cases"]))

    # --------------------------
    # Suggest New Rules
    # --------------------------
    st.subheader("Suggest New Rules (from Gold Test Results)")
    if st.button("💡 Suggest Rules from Failed Cases"):
        if "gold_summary" not in st.session_state:
            st.error("Run gold tests first!")
        else:
            gold_data = json.load(open(gold_path, "r", encoding="utf-8"))
            model_name = os.getenv("OLLAMA_MODEL", "deepseek-coder:6.7b")
            suggestions = suggest_rules_from_gold(gold_data, model_name=model_name, use_llm=use_llm)
            save_rule_suggestions(suggestions)
            st.session_state["suggestions"] = suggestions
            st.success(f"✅ {len(suggestions)} rule candidates generated.")
            st.dataframe(pd.DataFrame(suggestions))

    # --------------------------
    # Sanitize Suggested Rules
    # --------------------------
    st.subheader("🧹 Sanitize Rule Candidates")
    if st.button("🧪 Sanitize Rules"):
        if "suggestions" not in st.session_state:
            st.error("No rule suggestions found. Run 'Suggest Rules' first.")
        else:
            ok, msg, cleaned = sanitize_rules(st.session_state["suggestions"])
            st.info(msg)
            if ok:
                st.session_state["sanitized_rules"] = cleaned
                st.dataframe(pd.DataFrame(cleaned))
            else:
                st.warning("Some rules invalid — check the console log for details.")

    # --------------------------
    # Commit to GitHub
    # --------------------------
    st.subheader("📦 Commit Sanitized Rules to GitHub")

    gh_token = st.text_input("GitHub Token (stored securely in Streamlit Secrets)", type="password")
    repo_name = st.text_input("Repository (e.g., EphraimAsad/BactAI-D)", value=os.getenv("GITHUB_REPO", "EphraimAsad/BactAI-D"))

    # timestamped branch to prevent duplicates
    default_branch_name = f"bactai-learned-rules-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    new_branch_name = st.text_input("New branch name", value=default_branch_name)

    if st.button("⬆️ Commit Sanitized Rules"):
        if "sanitized_rules" not in st.session_state:
            st.error("No sanitized rules to commit.")
        elif not gh_token or not repo_name:
            st.error("Please enter GitHub token and repository.")
        else:
            try:
                g = Github(gh_token)
                repo = g.get_repo(repo_name)

                # Create new branch
                main_ref = repo.get_git_ref("heads/main")
                repo.create_git_ref(ref=f"refs/heads/{new_branch_name}", sha=main_ref.object.sha)

                # Write temp JSON
                tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".json").name
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump({"version": 1, "rules": st.session_state["sanitized_rules"]}, f, indent=2)

                # Commit
                with open(tmp_path, "r", encoding="utf-8") as f:
                    content = f.read()
                repo.create_file(
                    path="training/learned_rules.json",
                    message=f"feat(training): add {len(st.session_state['sanitized_rules'])} learned rules",
                    content=content,
                    branch=new_branch_name
                )

                # PR
                repo.create_pull(
                    title=f"Add learned rules ({datetime.now().strftime('%Y-%m-%d %H:%M')})",
                    body=f"Auto-generated from BactAI-D Streamlit UI with {len(st.session_state['sanitized_rules'])} rules.",
                    head=new_branch_name,
                    base="main"
                )

                st.success(f"✅ Committed {len(st.session_state['sanitized_rules'])} rules and opened PR.")
            except Exception as e:
                st.error(f"❌ GitHub commit failed: {e}")
