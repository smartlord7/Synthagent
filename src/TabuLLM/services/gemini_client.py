import os
import google.generativeai as genai
from dotenv import load_dotenv
from services.logger import log_interaction

load_dotenv()
genai.configure(api_key="AIzaSyA9k_qr9yyFLhZAWGboITSHLCYjIZlFaRA")
model = genai.GenerativeModel('gemini-2.5-flash')

def get_dataset_description(df_sample, provided_metadata=None):
    """
    Interprets or generates metadata for the dataset.
    """
    prompt = f"""
    Analyze this dataset sample and provide a concise summary of what this data represents.
    Metadata Provided: {provided_metadata if provided_metadata else "None"}
    Data Sample:
    {df_sample.to_string(index=False)}

    Return a 2-sentence description of the dataset context and the relationship between columns.
    """
    response = model.generate_content(prompt)
    log_interaction("Metadata Analysis", prompt, response.text)
    return response.text


def generate_data(target_column, context_data, dataset_context, all_columns, algorithm="Row-by-Row"):
    """
    Generates data using the semantic column names to ensure Gemini understands the context.
    """
    prompt = f"""
    Dataset Context: {dataset_context}

    You are an intelligent data entry assistant. 
    The dataset contains the following columns: {', '.join(all_columns)}

    Here is the data currently present in the target row:
    {context_data}

    Based on the context above, generate the most logical and realistic value for the column: '{target_column}'.
    If the row is entirely empty, invent a realistic, consistent value that fits the dataset.

    Return ONLY the value. No explanation, no quotes, no markdown.
    """

    try:
        response = model.generate_content(prompt)
        result = response.text.strip()
        log_interaction(algorithm, prompt, result)
        return result
    except Exception as e:
        print(f"Gemini API Error: {e}")
        log_interaction(algorithm, prompt, str(e), status="error")
        return ""