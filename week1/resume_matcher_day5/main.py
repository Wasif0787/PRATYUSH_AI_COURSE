import time
from pathlib import Path

from app.services.job_analyzer import analyze_job
from app.parsers.resume_parser import parse_resume
from app.services.matcher import match_resume
from app.utils.file_reader import read_resume
from pathlib import Path

JOB_DESCRIPTION_PATH = Path("data/job_description/jd.txt")
JOB_DESCRIPTION = JOB_DESCRIPTION_PATH.read_text(encoding="utf-8")


def main():

    # -------------------------
    # Analyze job
    # -------------------------

    job = analyze_job(JOB_DESCRIPTION)

    print("Job analyzed successfully")
    print("Minimum experience:", job.minimum_experience)

    # -------------------------
    # Process resumes
    # -------------------------

    resume_folder = Path("data/resumes")

    all_results = []

    for file_path in resume_folder.iterdir():

        if file_path.suffix.lower() not in [".pdf", ".docx"]:
            continue

        print(f"\nProcessing: {file_path.name}")

        resume_text = read_resume(file_path)

        parsed_resume = parse_resume(resume_text)

        time.sleep(5)

        result = match_resume(job, parsed_resume)

        time.sleep(5)

        print(f"Score: {result.score}%")

        all_results.append(
            {
                "name": parsed_resume.name,
                "score": result.score,
                "details": result.details,
            }
        )

    # -------------------------
    # Ranking
    # -------------------------

    all_results.sort(key=lambda candidate: candidate["score"], reverse=True)

    print("\nTOP 2 CANDIDATES")

    for candidate in all_results[:2]:

        print(candidate["name"], "-", candidate["score"], "%")

        print(candidate["details"])

    print("\nLOWEST 2 CANDIDATES")

    for candidate in all_results[-2:]:

        print(candidate["name"], "-", candidate["score"], "%")


if __name__ == "__main__":
    main()
