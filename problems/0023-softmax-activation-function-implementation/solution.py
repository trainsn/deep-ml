import math

def softmax(scores: list[float]) -> list[float]:
    # Your code here
    max_scores = max(scores)
    exps = [math.exp(s - max_scores) for s in scores]
    total = sum(exps)

    return [round(e / total, 4) for e in exps]
