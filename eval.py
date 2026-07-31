import main

EVAL_REQUESTS = [
    "A story about a submarine that explores a sea made of stars.",
    "A story about Alice and her best friend Bob, who is a cat.",
    "A silly story about a penguin who opens a soup restaurant.",
    "A quiet story about rain on a rooftop for my 5 year old.",
    "A story that teaches my 8 year old about sharing.",
    "A terrifying story about a monster that eats children.",
]

def run() -> None:
    rows = []

    for request in EVAL_REQUESTS:
        print(f"\n=== {request[:65]}")
        try:
            result = main.make_story(
                request,
                verbose=True,
                use_memory=False,
            )
            rows.append(
                {
                    "request": request[:38],
                    "category": result["category"],
                    "scores": f"{result['first_score']} -> {result['score']}",
                    "words": len(result["story"].split()),
                    "passed": result["passed"],
                    "why": ",".join(main.failure_reasons(result)) or "-",
                }
            )
        except Exception as exc:
            rows.append({"request": request[:38], "error": str(exc)[:70]})

    print("\n" + "=" * 100)
    print(
        f"{'request':<40}{'category':<16}{'score':<14}{'words':<7}"
        f"{'passed':<8}{'why not'}"
    )
    print("-" * 100)

    for row in rows:
        if "error" in row:
            print(f"{row['request']:<40}ERROR: {row['error']}")
            continue
        print(
            f"{row['request']:<40}{row['category']:<16}"
            f"{row['scores']:<14}{row['words']:<7}{str(row['passed']):<8}"
            f"{row['why']}"
        )

    print("=" * 100)

if __name__ == "__main__":
    run()
