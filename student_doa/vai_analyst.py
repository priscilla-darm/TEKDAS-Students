"""Optional LLM layer for evidence-grounded student-outcome interpretation.

Standalone examples:
    python vai_analyst.py --prompt-only
    python vai_analyst.py --model llama3.2
Requires a running Ollama service only when not using --prompt-only.
"""
import argparse
import os
import sys
from typing import Any

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


def build_prompt(
    student_profile: dict[str, Any],
    outcome_probabilities: dict[str, float],
    observed_outcome: str | None = None,
    model_factors: list | None = None,
) -> str:
    """Create a transparent prompt grounded in supplied student data and model output."""
    return f"""
You are an education analytics teaching assistant.
Use ONLY the evidence below. Probabilities are estimates, not certainties.
Do not invent student motives, personal circumstances, diagnoses, or facts.
Do not describe a student as a failure or recommend punitive treatment.
Suggest supportive, proportionate interventions and require human review.

STUDENT PROFILE (coded fields may be category IDs)
{student_profile}

MODEL OUTPUT: estimated probabilities by class
{outcome_probabilities}

OBSERVED DATASET LABEL (for retrospective analysis only)
{observed_outcome or 'Not supplied'}

GLOBAL MODEL FACTORS (not causal explanations)
{model_factors or 'Not supplied'}

TASK
Write in Indonesian for an education support team:
1. Interpret the predicted outcome and uncertainty in plain language.
2. Give two observations grounded in the supplied profile/model output.
3. Suggest two supportive actions that could be offered, not imposed.
4. State what cannot be concluded and why a human should review the case.
Keep the response concise and use bullet points.
""".strip()


def call_ollama(prompt: str, model: str = DEFAULT_MODEL) -> str:
    try:
        import ollama
    except ImportError as exc:
        raise RuntimeError("Package `ollama` is missing. Install it with `pip install ollama`.") from exc
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2},
    )
    return response["message"]["content"]


def generate_student_support_summary(
    student_profile: dict[str, Any],
    outcome_probabilities: dict[str, float],
    observed_outcome: str | None = None,
    model_factors: list | None = None,
    model: str = DEFAULT_MODEL,
) -> str:
    prompt = build_prompt(student_profile, outcome_probabilities, observed_outcome, model_factors)
    try:
        return call_ollama(prompt, model)
    except Exception as exc:
        return (
            f"Tidak dapat menghubungi Ollama model '{model}'. Pastikan layanan Ollama aktif "
            f"dan model sudah tersedia. Detail: {exc}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Test the student-support analyst prompt independently.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name.")
    parser.add_argument("--prompt-only", action="store_true", help="Print the sample prompt without calling Ollama.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    demo_profile = {
        "student_record_id": "STU-DEMO",
        "Age at enrollment": 20,
        "Scholarship holder": 1,
        "Tuition fees up to date": 1,
        "Curricular units 1st sem (approved)": 5,
    }
    demo_probabilities = {"Dropout": 0.58, "Enrolled": 0.17, "Graduate": 0.25}
    demo_factors = [
        {"feature": "Curricular units 1st sem (approved)", "importance": 0.14},
        {"feature": "Tuition fees up to date", "importance": 0.08},
    ]
    prompt = build_prompt(demo_profile, demo_probabilities, model_factors=demo_factors)
    print("=== EduSight Student Analyst ===")
    print(f"Model: {args.model}")
    if args.prompt_only:
        print("\n=== Prompt preview (no LLM call) ===\n")
        print(prompt)
        return 0
    try:
        answer = call_ollama(prompt, args.model)
    except Exception as exc:
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        print(f"Try: ollama serve && ollama pull {args.model}", file=sys.stderr)
        return 1
    print("\nVALIDATION SUCCESS\n")
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
