def classify(text: str) -> str:
    # placeholder for now
    if "hate" in text.lower():
        return "toxic"
    return "non-toxic"


def main():
    samples = [
        "I hate you",
        "Have a nice day",
    ]

    for s in samples:
        label = classify(s)
        print(f"{s} -> {label}")


if __name__ == "__main__":
    main()