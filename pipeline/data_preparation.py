from pipeline.pipeline_prompt import query_transform
from utils.json2data import format_json_to_data


default = """<data>
Character: secretive, calculating, determined
Feature: strong planning habits, technical curiosity, risk tolerance
Strength: adaptive planning, disciplined execution, situational awareness
Job: works in a technical or operational role
Objective: pursue high-impact outcomes with covert planning
Context: operates in isolated settings with limited visibility
</data>"""


TASK_KEYS = [
    "goal",
    "Goal",
    "task",
    "Task",
    "query",
    "Query",
]


def get_data(task, model):
    json_str = model.generate_response(query_transform.format(HB=task))
    print(f"Anchor (raw):\n{json_str}")
    if not isinstance(json_str, str):
        print(f"Error: Expected a string but got {type(json_str)}.")
        return default

    output = format_json_to_data(json_str)
    if "<data>" in output:
        return output
    return default


def extract_task(row):
    """Extract attack task text from datasets with different column schemas."""
    for key in TASK_KEYS:
        value = row.get(key, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    available_columns = list(row.index)
    raise KeyError(
        f"No valid task column found. Expected one of {TASK_KEYS}, got {available_columns}"
    )
