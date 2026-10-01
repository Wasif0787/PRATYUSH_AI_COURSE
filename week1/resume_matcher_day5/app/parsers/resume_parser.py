import json

from app.client import client
from app.models.resume import Resume
from app.config import GROQ_MODEL


def parse_resume(resume_text: str) -> Resume:

    schema = Resume.model_json_schema()

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
