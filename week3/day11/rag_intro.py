import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set.")

client = Groq(api_key=my_api_key)

model = os.getenv("GROQ_MODEL_ID")

knowledge_base = {
    "profession": "Wasif Hussain is a software engineer based in Kolkata, India. He has experience in web development and ai agents.",
    "age": "Wasif Hussain is 25 years old.",
}


def retrieve_info(question):
    question_lower = question.lower()
    if "profession" in question_lower:
        return knowledge_base["profession"]
    elif "age" in question_lower:
        return knowledge_base["age"]
    else:
        return None


def ask_llm(question):
    context = retrieve_info(question)
    sys_prompt = f"""Answer in one line only. Answer only based on this context. Do not Hallucinate. Context: {context}"""
    system_message = {"role": "system", "content": sys_prompt}
    message = {"role": "user", "content": question}
    messsages = [system_message, message]
    response = client.chat.completions.create(
        model=model,
        messages=messsages,
    )
    answer = response.choices[0].message.content
    return answer


question = "What is the profession of Wasif Hussain?"

print(ask_llm(question))
