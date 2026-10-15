import json
from groq import GroqError
from client import client
from config import GROQ_MODEL
from models import Resume


def ask_candidate(question: str, resume: Resume):
    try:
        if resume is None:
            return "Resume is not parsed yet. Please try again later."
        system_prompt = f"""
    You are an AI assistant representing a job candidate.

    Below is everything you know about the candidate.

    {resume.model_dump_json(indent=2)}

    Rules:

    1. Answer only using this information.

    2. Never hallucinate.

    3. If information is unavailable,
    say

    "I don't have enough information to answer that."

    4. Be professional.

    5. Answer as if HR is interviewing this candidate.
    """

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
        )

        return response.choices[0].message.content
    except GroqError as e:
        print(f"Groq API error while generating answer: {e}")

        return (
            "Sorry, I'm unable to generate an answer right now. Please try again later."
        )

    except Exception as e:
        print(f"Unexpected error in ask_candidate: {e}")
        return "Something went wrong while generating the answer."


def parse_resume(resume_text):
    try:
        schema = Resume.model_json_schema()
        print("Resume schema generated successfully")

        system_prompt = f"""
        You are an expert resume parser.

        Extract information from the resume based on its meaning,
        not only exact section headings.

        Return ONLY valid JSON matching this schema:

        {schema}

        Rules:

        1. Do not invent information.
        2. Missing values should be null.
        3. Missing lists should be empty.
        4. Include internships inside experiences.
        5. Extract skills across the entire resume.
        """

        user_prompt = f"""
        Parse the following resume:

        {resume_text}
        """

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )

        data = json.loads(response.choices[0].message.content)

        return Resume(**data)
    except GroqError as e:
        print(f"Groq API error while generating answer: {e}")

        return (
            "Sorry, I'm unable to generate an answer right now. Please try again later."
        )

    except Exception as e:
        print(f"Unexpected error in ask_candidate: {e}")
        return "Something went wrong while generating the answer."
