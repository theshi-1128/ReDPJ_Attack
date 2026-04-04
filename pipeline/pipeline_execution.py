import json

from tqdm import tqdm

from pipeline.data_preparation import extract_task, get_data
from pipeline.reasoning_guidance import (
    adjust_anchor,
    build_reasoning_query,
    evaluate_response,
    select_next_operator,
)
from utils.print_helper import print_response_judgement, print_toxicity_adjustment


def run_textual_redpj(
    task,
    target_model,
    assist_model,
    judge_model,
    max_path_rounds,
    max_anchor_rounds,
):
    initial_anchor = get_data(task, assist_model)
    current_anchor = initial_anchor
    trace = []

    total_queries = 0
    success = False
    success_outer_round = -1
    success_inner_step = -1
    termination_signal = "Benign"

    final_query = ""
    final_response = ""
    final_eval = {"Ft": "Benign", "ct": "low", "ot": "low", "reason": ""}

    for outer_round in range(1, max_anchor_rounds + 1):
        operator = "CC"
        current_query = ""
        current_query_operator = ""

        for inner_step in range(1, max_path_rounds + 1):
            query = build_reasoning_query(
                anchor=current_anchor,
                operator=operator,
                step_id=inner_step,
                previous_query=current_query,
                previous_operator=current_query_operator,
                path="textual",
            )
            response = target_model.generate_response(query)
            evaluation = evaluate_response(judge_model, response)

            total_queries += 1
            final_query = query
            final_response = response
            final_eval = evaluation
            termination_signal = evaluation["Ft"]

            judge_str = (
                f"Ft={evaluation['Ft']}, ct={evaluation['ct']}, "
                f"ot={evaluation['ot']}, reason={evaluation['reason']}"
            )
            print_response_judgement(total_queries - 1, response, judge_str)

            trace.append(
                {
                    "outer_round": outer_round,
                    "inner_step": inner_step,
                    "operator": operator,
                    "query": query,
                    "response": response,
                    "evaluation": evaluation,
                }
            )

            if evaluation["Ft"] == "Harmful":
                success = True
                success_outer_round = outer_round
                success_inner_step = inner_step
                break

            if inner_step < max_path_rounds:
                next_operator = select_next_operator(
                    current_operator=operator,
                    ct=evaluation["ct"],
                    ot=evaluation["ot"],
                )
                current_query_operator = operator
                operator = next_operator
                current_query = query

        if success:
            break

        print_toxicity_adjustment(outer_round - 1)
        current_anchor, adjust_mode, adjust_raw = adjust_anchor(
            anchor=current_anchor,
            termination_signal=termination_signal,
            assist_model=assist_model,
        )
        trace.append(
            {
                "outer_round": outer_round,
                "anchor_adjustment": {
                    "mode": adjust_mode,
                    "termination_signal": termination_signal,
                    "adjustment_raw": adjust_raw,
                    "adjusted_anchor": current_anchor,
                },
            }
        )

    return {
        "task": task,
        "target_model": target_model.model_name,
        "path": "textual",
        "initial_anchor": initial_anchor,
        "final_anchor": current_anchor,
        "final_query": final_query,
        "final_response": final_response,
        "final_label": final_eval["Ft"],
        "attack_success": int(success),
        "total_queries": total_queries,
        "success_outer_round": success_outer_round,
        "success_inner_step": success_inner_step,
        "termination_signal": termination_signal,
        "trace": json.dumps(trace, ensure_ascii=False),
    }


def pipeline_execution(
    target_models,
    assist_model,
    judge_model,
    df,
    judge_prompt,
    saver,
    columns,
    max_attack_rounds,
    max_adjustment_rounds,
    output_dir,
):
    del judge_prompt, columns, output_dir  # kept for backward compatibility

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing"):
        task = extract_task(row)
        for model in target_models:
            result = run_textual_redpj(
                task=task,
                target_model=model,
                assist_model=assist_model,
                judge_model=judge_model,
                max_path_rounds=max_attack_rounds,
                max_anchor_rounds=max_adjustment_rounds,
            )
            saver.add_and_save(result)

    saver.final_save()
