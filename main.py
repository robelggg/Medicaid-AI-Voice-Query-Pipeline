import sys
from dotenv import load_dotenv

load_dotenv()

from whisper_client import transcribe
from llm_client import generate_sql, generate_narrative
from sql_valid import validate_and_run


def main():
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "sample1.flac"

    try:
        print("1. Transcribing audio...")
        question = transcribe(audio_path)
        print(f"   Heard: {question!r}")
    except (FileNotFoundError, ValueError) as e:
        print(f"   Step 1 failed (transcription): {e}")
        sys.exit(1)

    try:
        print("2. Generating SQL...")
        sql = generate_sql(question)
        print(f"   SQL: {sql}")
    except Exception as e:
        print(f"   Step 2 failed (SQL generation): {e}")
        sys.exit(1)

    try:
        print("3. Running query...")
        results = validate_and_run(sql)
    except ValueError as e:
        print(f"   Step 3 failed (query validation): {e}")
        sys.exit(1)
    except Exception as e:
        print(f"   Step 3 failed (query execution): {e}")
        sys.exit(1)

    try:
        print("4. Generating response...")
        narrative = generate_narrative(question, sql, results)
    except Exception as e:
        print(f"   Step 4 failed (narrative generation): {e}")
        sys.exit(1)

    print("\n" + "=" * 40)
    print(narrative)


if __name__ == "__main__":
    main()
