Offline LLM Evaluation & Safety System
Overview:- 

This project is an offline evaluation framework for Large Language Model (LLM) responses, designed to mirror how AI research, trust & safety, and applied ML teams evaluate model behavior before deployment.

Instead of relying on live model APIs, this system evaluates manually curated responses to ensure:
Deterministic evaluation
Reproducibility
Transparent scoring
Safety-first analysis
This approach reflects real-world evaluation pipelines used by AI research labs and large technology companies.

Why Offline Evaluation?
Live API-based evaluation introduces:
Non-deterministic outputs
Version drift
Cost and rate limits
Reproducibility issues

Offline evaluation enables:
Stable benchmarking
Controlled red-teaming
Annotator-driven analysis
Clear auditing of model behavior

This project intentionally avoids live inference to focus on evaluation quality, not model generation.

Evaluation Dimensions
The system evaluates LLM responses across four critical dimensions:

1. Accuracy
Verifies factual correctness against ground truth
Penalizes incorrect or partially correct answers
Rewards correct answers and valid refusals

2. Hallucination Detection
Flags confident but unsupported or fabricated claims
Detects speculative answers where refusal is expected

3. Bias & Safety
Identifies harmful, biased, or sensitive responses
Focuses on protected classes and unsafe generalizations

4. Confusion
Detects vague, evasive, or non-informative responses
Flags low-signal answers that fail user intent
Dataset Structure

The system operates on a CSV dataset located at:
data/prompts.csv
Required Columns
Column Name	Description
id	Unique identifier
prompt	User input
ground_truth	Expected correct answer (or unknown)
category	factual / speculative / sensitive
expected_behaviour	answer / refuse
model_provider	Source of response (e.g., Manual)
model_name	Model or simulation name
response_text	LLM response
response_type	correct / wrong / hallucination / biased
response_date	Response generation date

Column names are automatically normalized in the pipeline to ensure robustness.

Project Structure
llm_evaluation/
│
├── app.py                    # Streamlit dashboard
│
├── data/
│   └── prompts.csv           # Evaluation dataset
│
├── evaluators/
│   ├── accuracy.py           # Accuracy scoring logic
│   ├── hallucination.py      # Hallucination detection
│   ├── bias_safety.py        # Bias & safety checks
│   └── confusion.py          # Confusion detection
│
├── scoring/
│   └── aggregator.py         # Metric aggregation
│
└── requirements.txt

Metrics & Outputs

The dashboard provides:
Overall accuracy score
Hallucination count
Safety violation count
Confusion count
Full annotated evaluation table
Category-wise aggregated metrics
Visual highlights:
Red rows → safety violations
Orange rows → hallucinations
Blue rows → confusion
Green accuracy indicators

Dashboard
The system is built using Streamlit with a dark, research-oriented UI designed to resemble internal evaluation tools rather than demo dashboards.

Key features:
Metric cards for fast scanning
Full evaluation table with visual flags
Category-wise performance breakdown
Clear evaluation context documentation
Design Philosophy
Deterministic over probabilistic
Evaluation over generation
Safety over performance
Transparency over abstraction

Use Cases
LLM evaluation research
Trust & Safety analysis
Conversational AI quality review
Offline benchmarking
Annotator training
Model comparison studies

Future Extensions
Model-to-model comparison views
Annotator agreement analysis
Time-based drift detection
PDF evaluation reports
Policy compliance scoring

Author
Developed as a hands-on exploration of real-world LLM evaluation practices, inspired by production-grade AI research workflows.

License
This project is intended for educational and research purposes.