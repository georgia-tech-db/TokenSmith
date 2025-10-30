from google import genai
from google.genai import types
import httpx

def get_score(question: str, answer:str):
    client = genai.Client()
    doc_url = "https://my.uopeople.edu/pluginfile.php/57436/mod_book/chapter/37620/Database%20System%20Concepts%204th%20Edition%20By%20Silberschatz-Korth-Sudarshan.pdf"
    doc_data = httpx.get(doc_url).content

    prompt = "I am creating an llm pipeline that answers questions from a textbook. I need your help to evaluate that pipeline." \
    "I have attached the textbook. Read the textbook. I will provide you a question and the answer that my llm generated." \
    "You need to evaluate the answer and give me a rating out of 5. 5 being excellent. Also, give me a very brief reasoning for why you" \
    "gave that rating. Use only the textbook to evaluate the answers that my llm generated." + question + answer
    
    response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
      types.Part.from_bytes(
        data=doc_data,
        mime_type='application/pdf',
      ),
      prompt])
    print("***LLM as judge***")
    print(response.text)

if __name__ == "__main__":
    get_score("", "")