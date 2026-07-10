import json

with open("outputs/20260709_172052_What_is_the_syntax_for_a_Python_lambda_f.json") as f:
    result = json.load(f)

print(result["model_response"].get("relevance_reasoning", "(not present)"))