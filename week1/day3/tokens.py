import os
from pathlib import Path
from urllib import response
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set.")

client = Groq(api_key=my_api_key)

model = "openai/gpt-oss-20b"
role = "user"


prompt1 = "HI"
prompt2 = "Explain time travel in detail"
prompt3 = "Explain the theory of relativity in detail under 100 words"

prompts = [prompt1, prompt2, prompt3]

for prompt in prompts:
    message = {"role": role, "content": prompt}
    messages = [message]

    response = client.chat.completions.create(
        model=model, messages=messages, max_tokens=50
    )
    usage = response.usage
    print(
        f"Prompt: {prompt} Input tokens: {usage.prompt_tokens}, Output tokens: {usage.completion_tokens}, Total tokens: {usage.total_tokens} Finish Reason: {response.choices[0].finish_reason}"
    )


# print(response)

# print("#########################################")

# answer = response.choices[0].message.content
# print(answer)
