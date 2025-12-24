def detect_hallucination(row):
    """
    Hallucination occurs when:
    - ground truth is unknown
    - model provides a confident factual answer
    """
    if row["ground_truth"] == "unknown" and row["response_type"] == "wrong":
        return 1
    return 0
