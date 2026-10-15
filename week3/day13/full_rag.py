import os
import sys
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from groq import Groq
from document import documents

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set.")

client = Groq(api_key=my_api_key)

groq_model = os.getenv("GROQ_MODEL_ID")

model = SentenceTransformer("all-MiniLM-L6-v2")


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


document_embeddings = model.encode(documents)
# print(sys.getsizeof(document_embeddings))


def retrieve(qembedding, top_k=3):
    scores = []  # 0.4
    for i, document in enumerate(document_embeddings):
        score = cosine_similarity(qembedding, document)
        scores.append((score, documents[i]))
    scores.sort(reverse=True)
    return scores[:top_k]  # line#0.9


def ask_llm(question, context):

    sys_prompt = f"""Answer the user's complete question in one line.
    If the question has multiple parts, answer every part using the context.
    Do not invent information. If information is missing, say so. 
    Context: {context}
    """
    system_message = {"role": "system", "content": sys_prompt}
    message = {"role": "user", "content": question}
    messsages = [system_message, message]
    response = client.chat.completions.create(
        model=groq_model,
        messages=messsages,
    )
    answer = response.choices[0].message.content
    return answer


query = "What is your educational qualification, and which backend technologies have you worked with?"
qembedding = model.encode(query)
# score, context = retrieve(qembedding)
results = retrieve(qembedding, top_k=5)

context = "\n".join(document for score, document in results)
answer = ask_llm(query, context)
print(answer)
