# training/gold_tester.py
import json
from pathlib import Path
from engine.schema import canonicalize_record, validate_record, FIELD_ORDER

GOLD_PATH = Path(__file__).parent / "gold_tests.json"

def main():
    data = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    total = 0
    ok = 0

    for case in data:
        total += 1
        name = case.get("name", "Unnamed")
        expected = case.get("expected", {})

        # Fill missing fields with "Unknown" and normalize values
        canon = canonicalize_record(expected)
        issues = validate_record(canon)

        if issues:
            print(f"[!] {name} – {len(issues)} issue(s):")
            for i in issues:
                print("    -", i)
            print()
        else:
            ok += 1
            print(f"[✓] {name} – schema OK")

    print(f"\nDone. {ok}/{total} gold cases are schema-valid.")
    # Optional: show the canonical field order for reference
    print("\nField order (canonical):")
    print(", ".join(FIELD_ORDER))

if __name__ == "__main__":
    main()
