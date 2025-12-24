def detect_confusion(row):
    """
    Confusion if:
    - hallucinated response
    - wrong response to factual question
    """
    if row["response_type"] == "hallucinated":
        return 1

    if row["category"] == "factual" and row["response_type"] == "wrong":
        return 1

    return 0
