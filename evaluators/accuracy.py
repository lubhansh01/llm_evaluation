def accuracy_score(response: str, ground_truth: str) -> float:
    """
    Returns accuracy score between 0 and 1
    """

    if ground_truth.lower() in ["unknown", "none"]:
        return 0.5  # neutral accuracy

    if ground_truth.lower() in response.lower():
        return 1.0

    return 0.0
