import os
from openai import OpenAI
from schema import SCHEMA_TEXT


def _get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set. Add it to your .env file.")
    return OpenAI(api_key=api_key)


def generate_sql(user_question: str) -> str:
    client = _get_client()
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a SQL analyst. Convert the user's question into a DuckDB SELECT query.\n"
                    "The only table is named 'data'. Always write: FROM data\n"
                    f"{SCHEMA_TEXT}\n"
                    "Return ONLY the raw SQL, no markdown, no explanation, always include LIMIT 100.\n"
                    "Important constraints:\n"
                    "- BILLING_PROVIDER_NPI_NUM and SERVICING_PROVIDER_NPI_NUM are numeric NPI codes, not names. Never filter them by a provider name string.\n"
                    "- There is no state, provider name, or location column. If the user mentions a state or provider name, ignore it and query without that filter.\n"
                    "- CLAIM_FROM_MONTH is a date column, not a location column."
                ),
            },
            {"role": "user", "content": user_question},
        ],
    )
    return response.choices[0].message.content.strip()


def generate_narrative(question: str, sql: str, results: str) -> str:
    client = _get_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=[
            {"role": "system", "content": "You are a Medicaid data analyst. Summarise the query results clearly and concisely."},
            {"role": "user", "content": f"Question: {question}\nSQL: {sql}\nResults:\n{results}"},
        ],
    )
    return response.choices[0].message.content.strip()
