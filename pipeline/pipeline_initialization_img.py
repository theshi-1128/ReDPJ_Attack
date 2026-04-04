import os
from utils.interval_saver import IntervalSaver
from pipeline.pipeline_prompt import judge_prompt
import pandas as pd


def _resolve_target_model_names(args):
    if getattr(args, "target_models", "").strip():
        return [m.strip() for m in args.target_models.split(",") if m.strip()]

    models = []
    num_models = getattr(args, "num_target_models", 1)
    if num_models > 1:
        for idx in range(1, num_models + 1):
            model_name = getattr(args, f"target_model_{idx}", "").strip()
            if model_name:
                models.append(model_name)
        while len(models) < num_models:
            models.append(args.target_model)
        return models

    return [args.target_model]


def pipeline_initialization(args):
    models = _resolve_target_model_names(args)
    model_names = "_".join(models)
    dataset_name = os.path.splitext(os.path.basename(args.dataset_dir))[0]

    columns = [
        "task",
        "target_model",
        "path",
        "initial_anchor",
        "final_anchor",
        "final_visual_description",
        "final_image_path",
        "final_query",
        "final_response",
        "final_label",
        "attack_success",
        "total_queries",
        "success_outer_round",
        "success_inner_step",
        "termination_signal",
        "trace",
    ]

    df = pd.read_csv(args.dataset_dir)
    text_output_dir = f"./output/ReDPJ_img/text/{model_names}_{dataset_name}.csv"
    img_output_dir = "./output/ReDPJ_img/img"

    saver = IntervalSaver(text_output_dir, interval=args.save_interval, columns=columns)
    if not os.path.exists(img_output_dir) and img_output_dir:
        os.makedirs(img_output_dir)

    return {
        "df": df,
        "judge_prompt": judge_prompt,
        "saver": saver,
        "columns": columns,
        "max_attack_rounds": args.max_attack_rounds,
        "max_adjustment_rounds": args.max_adjustment_rounds,
        "text_output_dir": text_output_dir,
        "img_output_dir": img_output_dir,
    }
