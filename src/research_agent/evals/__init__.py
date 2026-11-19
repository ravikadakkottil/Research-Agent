"""
Evals: measure how GOOD the results are (not whether the code runs — that's pytest).

    golden.py   load the answer key (evals/golden.lock.json) + match papers to it
    judge.py    LLM-as-judge: is each selected paper relevant to the question?
    metrics.py  score one run: recall, precision, recency quality, grounding
    runner.py   run topics x runs, aggregate, save a timestamped report

Command line: python evals/run_eval.py --help
"""
