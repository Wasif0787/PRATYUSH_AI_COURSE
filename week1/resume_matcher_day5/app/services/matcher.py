import json

from app.client import client
from app.models.job import JobD
from app.models.resume import Resume
from app.models.match import MatchResult
from app.config import GROQ_MODEL


def match_resume(job: JobD, resume: Resume) -> MatchResult:

    schema = MatchResult.model_json_schema()

    prompt = f"""
    You are an HR recruiter.

    Compare the candidate's resume with the job description.

    JOB DESCRIPTION:
    {job.model_dump_json(indent=2)}

    CANDIDATE RESUME:
    {resume.model_dump_json(indent=2)}

    Return JSON matching this schema:

    {schema}

    Give:

    1. Candidate name
    2. Matching skills
    3. Missing important skills
    4. Whether experience requirement is met
    5. Overall match percentage from 0 to 100
    6. Short final verdict

    Keep the response concise.
    """

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    data = json.loads(response.choices[0].message.content)

    return MatchResult(**data)
