import json

from app.client import client
from app.models.job import JobD
from app.config import GROQ_MODEL


def analyze_job(job_description: str) -> JobD:

    schema = JobD.model_json_schema()

    system_prompt = f"""
    You are an expert HR assistant.

    Extract structured information from the job description.

    Return ONLY valid JSON matching this schema:

    {schema}

    Do not return the schema itself.

    If minimum experience is not mentioned, return null.

    If information for a list is missing,
    return an empty list.

    Do not invent information.
    """

    user_prompt = f"""
    Analyze the following job description:

    {job_description}
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

    return JobD(**data)
