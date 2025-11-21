import streamlit as st
import pandas as pd
import sqlparse
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import re

# ===========================
# Streamlit config
# ===========================
st.set_page_config(
    page_title="SML Qwen3-0.6B Text2SQL Demo",
    page_icon="🧠",
    layout="wide",
)

# ===========================
# Helpers: dataset loader
# ===========================
@st.cache_data
def load_single_table_df(path: str):
    df = pd.read_parquet(path)
    return df

@st.cache_data
def load_bird_df(path: str):
    df = pd.read_parquet(path)
    return df

# ===========================
# Helper: format SQL
# ===========================
def pretty_sql(sql: str) -> str:
    try:
        return sqlparse.format(sql, reindent=True, keyword_case="upper")
    except Exception:
        return sql


# ===========================
# Helper: xử lý "thinking" & chat prompts
# ===========================

def remove_think_block(text: str) -> str:
    """
    Loại bỏ phần reasoning nội bộ (thường nằm trong <think>...</think> hoặc
    <|begin_of_thought|>...</|end_of_thought|>), chỉ giữ lại phần SQL cuối.
    """
    patterns = [
        (r"<think>.*?</think>", re.DOTALL),
        (r"<\|begin_of_thought\|>.*?<\|end_of_thought\|>", re.DOTALL),
    ]
    out = text
    for pattern, flags in patterns:
        out = re.sub(pattern, "", out, flags=flags)
    return out.strip()


def build_single_table_prompts(row: pd.Series):
    """
    Tương đương với predictSingleTable() trên Colab:
    - system_prompt cố định
    - user_prompt = Schema + User Query + 'SQL Query:'
    """
    system_prompt = (
        "Write a SQL statement that is equivalent to the natural language user query below. "
        'You are given the schema in the format of a CREATE TABLE SQL statement. '
        'Assume the table is called "df". DO NOT give any preamble or extra characters or markdown '
        "just the SQL query in plain text. Make sure the SQL query is on one line."
    )

    schema = row.get("schema", "")
    query = row.get("query", "")

    user_prompt = f"Schema:\n{schema}\n\nUser Query:\n{query}\n\nSQL Query:\n"

    return system_prompt, user_prompt


def build_bird_prompts(row: pd.Series):
    """
    Tương đương với predictBIRD() trên Colab.
    Lưu ý: parquet BIRD có thể dùng 'question' hoặc bạn đã map sang 'query'.
    """
    system_prompt = """Task Overview:
You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.

Database Engine:
SQLite"""

    schema = row.get("schema", row.get("db_schema", ""))
    question = row.get("query", row.get("question", ""))  # fallback: query -> question

    user_prompt = f"""Database Schema:
{schema}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
{question}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- Do NOT hallucinate: only use tables, columns, and values that exist in the provided schema.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query.
- Keep the SQL minimal: no extra joins, filters, grouping, ordering, or aliases unless required.
- DO NOT give any preamble or extra characters or markdown, just the SQL query in plain text on a single line with no line breaks or indentation. DO NOT use any code fences or the substring ```sql in the output.
- Take a deep breath and think step by step to find the correct SQL query.

SQL Query:"""

    return system_prompt, user_prompt


def serialize_chat(system_prompt: str, user_prompt: str, tokenizer, enable_thinking: bool) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return text


# ===========================
# HF model loader & generator
# ===========================

@st.cache_resource
def load_hf_model_and_tokenizer(model_name: str):
    """
    Load tokenizer & model HF, cache lại để không reload mỗi lần click.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    return tokenizer, model, device


def generate_single_sql(row: pd.Series, model_name: str, max_new_tokens: int = 1024):
    """
    Bản Streamlit tương đương predictSingleTable(...) trên Colab.
    Trả về: (serialized_prompt, response_sql)
    """
    tokenizer, model, device = load_hf_model_and_tokenizer(model_name)
    system_prompt, user_prompt = build_single_table_prompts(row)
    text = serialize_chat(system_prompt, user_prompt, tokenizer, enable_thinking=False)

    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
        )

    # cắt phần input, chỉ lấy phần model sinh thêm
    gen_ids = output_ids[:, model_inputs["input_ids"].shape[1]:]
    response = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

    return text, response


def generate_bird_sql(row: pd.Series, model_name: str, max_new_tokens: int = 1024):
    """
    Bản Streamlit tương đương predictBIRD(...) trên Colab.
    Trả về: (serialized_prompt, response_sql) sau khi remove_think_block.
    """
    tokenizer, model, device = load_hf_model_and_tokenizer(model_name)
    system_prompt, user_prompt = build_bird_prompts(row)
    text = serialize_chat(system_prompt, user_prompt, tokenizer, enable_thinking=True)

    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_ids = output_ids[:, model_inputs["input_ids"].shape[1]:]
    raw_response = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
    cleaned_response = remove_think_block(raw_response).strip()

    return text, cleaned_response

# ===========================
# Main UI
# ===========================
def main():
    st.title("🧠 SML Qwen3-0.6B Text2SQL Demo")
    st.caption("Supervised finetune • Text-to-SQL • Dev evaluation UI (HF models)")

    st.sidebar.markdown(
        """
    <div style="
        padding: 0.75rem 0.9rem;
        border-radius: 0.75rem;
        background-color: #f7f7f9;
        border: 1px solid #e0e0e0;
        font-size: 0.9rem;
    ">
    <div style="font-weight: 600; margin-bottom: 0.25rem; font-size: 0.95rem;">
        🧑‍💻 Thông tin nhóm
    </div>
    <div style="margin-bottom: 0.5rem;">
        <div><strong>Môn học:</strong> Hệ Cơ Sở Dữ Liệu Tiên Tiến</div>
        <div><strong>Mã môn học:</strong> IS6002.CH191</div>
    </div>

    <div style="font-weight: 600; margin-bottom: 0.25rem;">
        Team 10
    </div>
    <ul style="padding-left: 1.1rem; margin: 0;">
        <li><strong>Võ Phạm Duy Đức</strong> · MSSV: <em>240104028</em></li>
        <li><strong>Đỗ Trọng Khánh</strong> · MSSV: <em>240104036</em></li>
        <li><strong>Trần Bảo Ân</strong> · MSSV: <em>240104024</em></li>
    </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")

    # === Cấu hình đường dẫn dataset ===
    st.sidebar.header("⚙️ Cấu hình dữ liệu")

    default_single_path = "data/single-table_dev.parquet"
    default_bird_path = "data/bird_dev.parquet"

    single_path = st.sidebar.text_input(
        "Đường dẫn Single-Table dev parquet",
        value=default_single_path,
    )
    bird_path = st.sidebar.text_input(
        "Đường dẫn BIRD dev parquet",
        value=default_bird_path,
    )

    # === Cấu hình model HF ===
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 Cấu hình HuggingFace models")

    default_single_model = os.getenv("HF_MODEL_SINGLE", "ZiDuck/Qwen3-0.6B-Text2SQL")
    default_bird_model = os.getenv("HF_MODEL_BIRD", "ZiDuck/SFT-Qwen3-0.6B-Text2SQL-MiniBIRD")

    single_model_name = st.sidebar.text_input(
        "HF model cho Single-Table",
        value=default_single_model,
        help="VD: ZiDuck/Qwen3-0.6B-Text2SQL",
    )

    bird_model_name = st.sidebar.text_input(
        "HF model cho BIRD mini_dev",
        value=default_bird_model,
        help="VD: ZiDuck/SFT-Qwen3-0.6B-Text2SQL-MiniBIRD",
    )


    # Tabs cho 2 bộ dữ liệu
    tab_single, tab_bird = st.tabs(["📄 Single-Table", "🦅 BIRD mini_dev"])

    # ===========================
    # Tab 1: Single-Table
    # ===========================
    with tab_single:
        st.subheader("Single-Table Dev Dataset")

        # Load df
        try:
            df_single = load_single_table_df(single_path)
        except Exception as e:
            st.error(f"Không load được file: {single_path}")
            st.exception(e)
            df_single = None

        if df_single is not None and not df_single.empty:
            st.write(f"Số mẫu dev: **{len(df_single)}**")

            # Hiển thị bảng (chỉ 200 record đầu cho nhẹ)
            display_cols = [c for c in ["query", "schema", "sql", "source"] if c in df_single.columns]
            st.dataframe(
                df_single[display_cols].head(200),
                use_container_width=True,
                height=400,
            )

            st.markdown("### Chọn 1 câu hỏi để dự đoán")

            options = []
            for idx, row in df_single.iterrows():
                q = row.get("query", "")
                q_str = "" if q is None else str(q)
                preview = q_str[:120] + ("..." if len(q_str) > 120 else "")
                options.append(f"{idx}: {preview}")

            selected_option = st.selectbox(
                "Chọn sample (index: nội dung câu hỏi)",
                options=options,
                index=0,
                key="single_select",
            )

            selected_idx = int(selected_option.split(":")[0])
            selected_row = df_single.iloc[selected_idx]

            # ===== Bước 1: Thông tin record =====
            st.markdown("## Bước 1️⃣: Thông tin record đã chọn")
            with st.expander("Xem chi tiết record", expanded=True):
                st.write("**User question:**")
                st.write(selected_row.get("query", ""))

                st.write("---")
                st.write("**Schema:**")
                st.code(str(selected_row.get("schema", "")), language="sql")

                st.write("---")
                st.write("**Ground truth SQL:**")
                st.code(pretty_sql(str(selected_row.get("sql", ""))), language="sql")

                st.write("---")
                st.write("**Source:** ", selected_row.get("source", ""))

            # ===== Bước 2: Prompt gửi vào LLM =====
            st.markdown("## Bước 2️⃣: Prompt gửi vào LLM")

            system_prompt_single, user_prompt_single = build_single_table_prompts(selected_row)

            with st.expander("System prompt", expanded=True):
                st.code(system_prompt_single, language="markdown")

            with st.expander("User prompt", expanded=True):
                st.code(user_prompt_single, language="markdown")

            # (Tuỳ chọn) Hiển thị luôn chuỗi serialized sau chat_template
            with st.expander("Chuỗi đầu vào thực tế", expanded=False):
                tokenizer_single, _, _ = load_hf_model_and_tokenizer(single_model_name)
                serialized_single = serialize_chat(
                    system_prompt_single,
                    user_prompt_single,
                    tokenizer_single,
                    enable_thinking=False,
                )
                st.code(serialized_single, language="text")


            # ===== Bước 3: Dự đoán SQL =====
            st.markdown("## Bước 3️⃣: Kết quả SQL dự đoán")

            run_single = st.button("🚀 Thực thi Text2SQL (Single-Table)", key="run_single")

            if run_single:
                with st.spinner(f"Đang gọi HF model `{single_model_name}` để sinh SQL..."):
                    serialized_prompt_single, predicted_sql = generate_single_sql(
                        selected_row,
                        model_name=single_model_name,
                        max_new_tokens=1024,
                    )

                st.success("Đã sinh xong SQL.")
                st.markdown("**SQL dự đoán:**")
                st.code(pretty_sql(predicted_sql), language="sql")

                # Thêm Gold SQL
                st.markdown("**Ground Truth SQL (Gold):**")
                st.code(pretty_sql(str(selected_row.get("sql", ""))), language="sql")

        else:
            st.info("Chưa có dữ liệu Single-Table hoặc load lỗi.")

    # ===========================
    # Tab 2: BIRD
    # ===========================
    with tab_bird:
        st.subheader("BIRD mini_dev Dataset")

        try:
            df_bird = load_bird_df(bird_path)
        except Exception as e:
            st.error(f"Không load được file: {bird_path}")
            st.exception(e)
            df_bird = None

        if df_bird is not None and not df_bird.empty:
            st.write(f"Số mẫu dev: **{len(df_bird)}**")

            display_cols = [c for c in ["db_id", "question", "evidence", "SQL"] if c in df_bird.columns]
            st.dataframe(
                df_bird[display_cols].head(200),
                use_container_width=True,
                height=400,
            )

            st.markdown("### Chọn 1 câu hỏi để dự đoán")
            options_bird = []
            for idx, row in df_bird.iterrows():
                db_id = row.get("db_id", "")
                q = row.get("question", "")
                q_str = "" if q is None else str(q)
                preview = q_str[:120] + ("..." if len(q_str) > 120 else "")
                options_bird.append(f"{idx} | db: {db_id} | {preview}")

            selected_option_bird = st.selectbox(
                "Chọn sample (index | db_id | question)",
                options=options_bird,
                index=0,
                key="bird_select",
            )

            selected_idx_bird = int(selected_option_bird.split("|")[0].strip())
            selected_row_bird = df_bird.iloc[selected_idx_bird]

            # ===== Bước 1: Thông tin record =====
            st.markdown("## Bước 1️⃣: Thông tin record đã chọn")
            with st.expander("Xem chi tiết record", expanded=True):
                st.write("**DB ID:** ", selected_row_bird.get("db_id", ""))

                st.write("---")
                st.write("**User question:**")
                st.write(selected_row_bird.get("question", ""))

                st.write("---")
                st.write("**Evidence (domain knowledge):**")
                st.write(selected_row_bird.get("evidence", ""))

                st.write("---")
                schema_value = selected_row_bird.get("schema", selected_row_bird.get("db_schema", ""))
                if schema_value != "":
                    st.write("**Schema:**")
                    st.code(str(schema_value), language="sql")
                else:
                    st.info("Parquet BIRD hiện không chứa cột schema/db_schema. Prompt vẫn hoạt động nhưng thiếu schema.")

                st.write("---")
                st.write("**Ground truth SQL:**")
                st.code(pretty_sql(str(selected_row_bird.get("SQL", ""))), language="sql")

            # ===== Bước 2: Prompt gửi vào LLM =====
            st.markdown("## Bước 2️⃣: Prompt gửi vào LLM")

            system_prompt_bird, user_prompt_bird = build_bird_prompts(selected_row_bird)

            with st.expander("System prompt", expanded=True):
                st.code(system_prompt_bird, language="markdown")

            with st.expander("User prompt", expanded=True):
                st.code(user_prompt_bird, language="markdown")

            with st.expander("Chuỗi đầu vào thực tế (sau chat_template)", expanded=False):
                tokenizer_bird, _, _ = load_hf_model_and_tokenizer(bird_model_name)
                serialized_bird = serialize_chat(
                    system_prompt_bird,
                    user_prompt_bird,
                    tokenizer_bird,
                    enable_thinking=True,
                )
                st.code(serialized_bird, language="text")

            # ===== Bước 3: Dự đoán SQL =====
            st.markdown("## Bước 3️⃣: Kết quả SQL dự đoán")
            run_bird = st.button("🚀 Thực thi Text2SQL (BIRD)", key="run_bird")

            if run_bird:
                with st.spinner(f"Đang gọi HF model `{bird_model_name}` để sinh SQL..."):
                    serialized_prompt_bird, predicted_sql_bird = generate_bird_sql(
                        selected_row_bird,
                        model_name=bird_model_name,
                        max_new_tokens=1024,
                    )

                st.success("Đã sinh xong SQL.")
                st.markdown("**SQL dự đoán:**")
                st.code(pretty_sql(predicted_sql_bird), language="sql")

                # Thêm Gold SQL (cột BIRD thường là "SQL")
                st.markdown("**Ground Truth SQL (Gold):**")
                st.code(pretty_sql(str(selected_row_bird.get("SQL", ""))), language="sql")

        else:
            st.info("Chưa có dữ liệu BIRD hoặc load lỗi.")


if __name__ == "__main__":
    main()
