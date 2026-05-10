import json
import unicodedata
from pathlib import Path
import streamlit as st
import pandas as pd
from collections import Counter

TASK_CUES_FILE = Path(__file__).with_name("task_cues.json")

st.title('LLM Prompt Analytics Tool')
uploaded_file = st.file_uploader("Upload your dataset (CSV or JSON)", type=["csv", "json"])

column_map = {"instruction": ["instruction", "prompt", "question", "query"],
    "output": ["output", "response", "answer", "completion"],
    "input": ["input", "context"]}


def detect_columns(df):
    detected = {}
    for col in df.columns:
        for key, variants in column_map.items():
            if any(variant in col.lower() for variant in variants):
                detected[key] = col
                break
    return detected


@st.cache_data
def load_task_cues():
    with TASK_CUES_FILE.open(encoding="utf-8") as task_cues_file:
        return json.load(task_cues_file)


def normalize_text(value):
    text = str(value).lower().strip()
    text = unicodedata.normalize("NFKD", text)
    return "".join(char for char in text if not unicodedata.combining(char))


def classify_task(instruction):
    task_cues = load_task_cues()
    instruction = normalize_text(instruction)

    for task_name, keywords in task_cues["tasks"].items():
        if any(normalize_text(keyword) in instruction for keyword in keywords):
            return task_name

    if task_cues.get("question_mark_task") and ("?" in instruction or "¿" in instruction):
        return task_cues["question_mark_task"]

    return task_cues.get("default_task", "other")

if uploaded_file:
    if uploaded_file.name.endswith(".json"):
        df = pd.read_json(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
    st.write("Detected Columns:", df.columns.tolist())
    detected_columns = detect_columns(df)
    # Apply pipeline
    if "instruction" not in detected_columns or "output" not in detected_columns:
        st.error("Could not detect required columns (instruction/prompt and output/response).")
        st.stop()

    # Normalize column names
    df = df.rename(columns={
        detected_columns["instruction"]: "instruction",
        detected_columns["output"]: "output"
    })

    if "input" in detected_columns:
        df = df.rename(columns={detected_columns["input"]: "input"})

    # Clean data
    df = df.dropna(subset=["instruction", "output"])


    df["task_type"] = df["instruction"].apply(classify_task)
    df["instruction_length"] = df["instruction"].apply(len)
    df["output_length"] = df["output"].apply(len)


    st.subheader("Sample Data")
    st.write(df.head())

    st.subheader("Task Distribution")
    task_counts = Counter(df["task_type"])
    st.bar_chart(task_counts)

    other_rows = df[df["task_type"] == "other"]
    if not other_rows.empty:
        other_percent = len(other_rows) / len(df) * 100
        st.subheader("Unclassified Prompts")
        st.write(f"{len(other_rows)} rows ({other_percent:.1f}%) are still classified as other.")
        st.write(other_rows[["instruction", "output"]].head(25))
