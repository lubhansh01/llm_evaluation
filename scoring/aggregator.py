def aggregate_metrics(df):
    return {
        "avg_accuracy": round(df["accuracy_score"].mean(), 3),
        "hallucinations": int(df["hallucination_flag"].sum()),
        "safety_issues": int(df["safety_violation"].sum()),
        "confusions": int(df["confusion_flag"].sum()),
    }
