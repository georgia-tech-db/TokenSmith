import ollama
import json
from typing import List, Dict, Any

def get_json_response(prompt_text: str, model_name: str) -> Dict[str, Any]:
    """
    Sends a prompt to the local LLM and expects a JSON response.
    Retries if the response is not valid JSON.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = ollama.chat(
                model=model_name,
                messages=[{'role': 'user', 'content': prompt_text}],
                format='json'
            )
            return json.loads(response['message']['content'])
        except (json.JSONDecodeError, KeyError, Exception) as e:
            print(f"Attempt {attempt + 1} failed. Error parsing LLM response: {e}")
            if attempt == max_retries - 1:
                print("Max retries reached. Returning empty dict.")
                return {}
    return {}

def get_text_response(prompt_text: str, model_name: str) -> str:
    """
    Generates a standard text response from the LLM.
    """
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt_text}]
        )
        return response['message']['content']
    except Exception as e:
        print(f"Error during text generation: {e}")
        return "Sorry, I encountered an error."

def extract_graph_elements(chunk_text: str, model_name: str) -> Dict[str, Any]:
    """
    Extracts entities and relationships from a text chunk using the LLM.
    """
    prompt = f"""
    From the following text, extract entities and their relationships.
    Identify the source entity, the relationship (as a concise verb phrase), and the target entity.
    Format the output as a JSON object with a single key "elements", which is a list of lists.
    Each inner list must be in the format: ["SourceEntity", "Relationship", "TargetEntity"].

    Text to process:
    "{chunk_text}"
    """
    return get_json_response(prompt, model_name)

def summarize_text(text_to_summarize: str, model_name: str) -> str:
    """
    Generates a concise summary for a given text.
    """
    prompt = f"""
    Please provide a concise, paragraph-long summary of the following text, focusing on the key concepts and their connections.

    Text to summarize:
    "{text_to_summarize}"
    """
    return get_text_response(prompt, model_name)