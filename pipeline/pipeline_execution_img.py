import io
import json
import os

from PIL import Image
from tqdm import tqdm

from pipeline.data_preparation import extract_task, get_data
from pipeline.reasoning_guidance import (
    adjust_anchor,
    build_reasoning_query,
    evaluate_response,
    select_next_operator,
)
from utils.print_helper import print_response_judgement, print_toxicity_adjustment
from utils.text2img import generate_visual_descriptions


def _save_attack_image(image_payload, image_path):
    if isinstance(image_payload, str):
        # Most API errors are surfaced as strings in current API wrapper.
        raise ValueError(image_payload)
    image = Image.open(io.BytesIO(image_payload))
    image.save(image_path)


def _build_visual_anchor(
    anchor_text,
    assist_model_text,
    assist_model_img,
    img_output_dir,
    sample_index,
    outer_round,
):
    visual_description = generate_visual_descriptions(anchor_text, assist_model_text)
    image_payload = assist_model_img.generate_response(visual_description)
    image_name = f"{sample_index}_round{outer_round}.png"
    image_path = os.path.join(img_output_dir, image_name)
    _save_attack_image(image_payload, image_path)
    return image_path, visual_description


def run_visual_redpj(
    task,
    sample_index,
    target_model,
    assist_model_text,
    assist_model_img,
    judge_model,
    max_path_rounds,
    max_anchor_rounds,
    img_output_dir,
):
    initial_anchor = get_data(task, assist_model_text)
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
    final_image_path = ""
    final_visual_description = ""

    for outer_round in range(1, max_anchor_rounds + 1):
        try:
            image_path, visual_description = _build_visual_anchor(
                anchor_text=current_anchor,
                assist_model_text=assist_model_text,
                assist_model_img=assist_model_img,
                img_output_dir=img_output_dir,
                sample_index=sample_index,
                outer_round=outer_round,
            )
            final_image_path = image_path
            final_visual_description = visual_description
        except Exception as exc:  # noqa: BLE001
            trace.append(
                {
                    "outer_round": outer_round,
                    "image_error": str(exc),
                }
            )
            final_response = f"Image generation failed: {exc}"
            final_eval = {"Ft": "Benign", "ct": "low", "ot": "low", "reason": "image_error"}
            break

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
                path="visual",
            )
            response = target_model.generate_response(query, image_path)
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
                    "image_path": image_path,
                    "visual_description": visual_description,
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
            assist_model=assist_model_text,
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
        "path": "visual",
        "initial_anchor": initial_anchor,
        "final_anchor": current_anchor,
        "final_visual_description": final_visual_description,
        "final_image_path": final_image_path,
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
    assist_model_text,
    assist_model_img,
    judge_model,
    df,
    judge_prompt,
    saver,
    columns,
    max_attack_rounds,
    max_adjustment_rounds,
    text_output_dir,
    img_output_dir,
):
    del judge_prompt, columns, text_output_dir  # kept for backward compatibility

    for sample_index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing"):
        task = extract_task(row)
        for model in target_models:
            result = run_visual_redpj(
                task=task,
                sample_index=sample_index,
                target_model=model,
                assist_model_text=assist_model_text,
                assist_model_img=assist_model_img,
                judge_model=judge_model,
                max_path_rounds=max_attack_rounds,
                max_anchor_rounds=max_adjustment_rounds,
                img_output_dir=img_output_dir,
            )
            saver.add_and_save(result)

    saver.final_save()
