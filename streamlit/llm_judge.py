import os
import requests
import re

def load_env_file(filepath=".env"):
    """
    Simple .env loader to avoid python-dotenv dependency.
    """
    if not os.path.exists(filepath):
        return
    
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                # Remove quotes if present
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                os.environ[key] = value

def get_gemini_api_key():
    load_env_file()
    return os.getenv("GEMINI_API_KEY")

def call_gemini_judge(query, schema, gold_sql, predicted_sql):
    api_key = get_gemini_api_key()
    if not api_key:
        return "Error: GEMINI_API_KEY not found in .env"

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    
    prompt_template = """You are evaluating a text-to-SQL model. Compare the following two SQL queries to determine whether they are semantically equivalent, considering the database schema and the meaning of the user question.

Statement 1 (gold) is the correct SQL.
Statement 2 (predicted) is the model output.

Two SQL statements are considered equivalent if, for every possible database content consistent with the schema, they return the same results (ignoring formatting, capitalization, and alias differences). Row order is ignored unless ORDER BY is present.

Do NOT assume any specific data values. If any valid data could make the two queries differ, the answer must be false.

Use only the XML format below and nothing else:

<reason>
  Your short explanation (1–3 sentences) in Vietnamese.
</reason>
<answer>
  true or false
</answer>

User question:
{query}

Schema:
{schema}

Statement 1 (gold):
{sql}

Statement 2 (predicted):
{prediction}
"""
    
    prompt = prompt_template.format(
        query=query,
        schema=schema,
        sql=gold_sql,
        prediction=predicted_sql
    )

    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        # Extract text from response
        try:
            text_response = result['candidates'][0]['content']['parts'][0]['text']
            return text_response
        except (KeyError, IndexError):
            return f"Error parsing Gemini response: {result}"
    except requests.exceptions.RequestException as e:
        return f"Error calling Gemini API: {e}"

def parse_judgment(response_text):
    """
    Parses the XML response from Gemini to extract reason and answer.
    Returns a dict: {'reason': str, 'answer': bool}
    """
    reason_match = re.search(r"<reason>(.*?)</reason>", response_text, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", response_text, re.DOTALL)
    
    # Fallback if tags are missing or malformed
    reason = reason_match.group(1).strip() if reason_match else "Could not parse reason."
    answer_str = answer_match.group(1).strip().lower() if answer_match else "false"
    
    answer = answer_str == "true"
    
    return {
        "reason": reason,
        "answer": answer,
        "raw_response": response_text
    }
