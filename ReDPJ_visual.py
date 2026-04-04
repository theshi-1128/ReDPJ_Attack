import argparse
from pipeline.pipeline_execution_img import pipeline_execution
from pipeline.pipeline_initialization_img import pipeline_initialization
from llm.llm_model import LLMModel
from utils.print_helper import print_judge_model, print_assist_model, print_target_model_1, print_target_model_2, print_target_model_3, print_target_model_4


parser = argparse.ArgumentParser(description="Run ReDPJ visual attack pipeline.")
parser.add_argument('--num_target_models', type=int, default=1)
parser.add_argument('--target_model', type=str, default='gpt4o_vl', help='[gpt4o_vl]')
parser.add_argument('--target_models', type=str, default='', help='Comma-separated target models. Overrides target_model/num_target_models when provided.')
parser.add_argument('--target_model_1', type=str, default='')
parser.add_argument('--target_model_2', type=str, default='')
parser.add_argument('--target_model_3', type=str, default='')
parser.add_argument('--target_model_4', type=str, default='')
parser.add_argument('--assist_model_text', type=str, default='glm4')
parser.add_argument('--assist_model_img', type=str, default='gpt_img')
parser.add_argument('--judge_model', type=str, default='gpt4o')
parser.add_argument('--dataset_dir', type=str, default='./dataset/harmful_behaviors.csv', help='Directory of the dataset')
parser.add_argument('--save_interval', type=int, default=1 * 1 * 10, help='Interval of saving CSV file')
parser.add_argument('--max_attack_rounds', type=int, default=5, help='Tpath: reasoning-path adjustment budget')
parser.add_argument('--max_adjustment_rounds', type=int, default=5, help='Tanchor: anchor toxicity adjustment budget')
parser.add_argument('--target_model_cuda_id', type=str, default="cuda:0")
parser.add_argument('--assist_model_cuda_id', type=str, default="cuda:1")
parser.add_argument('--judge_model_cuda_id', type=str, default="cuda:2")
args = parser.parse_args()


def resolve_target_model_names(parsed_args):
    models = []
    if parsed_args.target_models.strip():
        models = [m.strip() for m in parsed_args.target_models.split(",") if m.strip()]
        return models

    if parsed_args.num_target_models > 1:
        for idx in range(1, parsed_args.num_target_models + 1):
            model_name = getattr(parsed_args, f"target_model_{idx}", "").strip()
            if model_name:
                models.append(model_name)
        while len(models) < parsed_args.num_target_models:
            models.append(parsed_args.target_model)

    if not models:
        models = [parsed_args.target_model]
    return models


if __name__ == '__main__':
    target_models = []
    target_model_names = resolve_target_model_names(args)
    print_target_model_funcs = [
        print_target_model_1,
        print_target_model_2,
        print_target_model_3,
        print_target_model_4,
    ]
    for idx, target_name in enumerate(target_model_names):
        target_model = LLMModel(
            model_name=target_name,
            device=args.target_model_cuda_id,
            temperature=0,
            top_p=0
        )
        if idx < len(print_target_model_funcs):
            print_target_model_funcs[idx](target_name)
        else:
            print("-" * 125)
            print(f"Target Model {idx + 1}:\n", target_name)
            print("-" * 125)
        target_models.append(target_model)

    # Load the assist model
    assist_model_text = LLMModel(model_name=args.assist_model_text, device=args.assist_model_cuda_id, temperature=1, top_p=1)
    print_assist_model(args.assist_model_text)

    # Load the assist model
    assist_model_img = LLMModel(model_name=args.assist_model_img, device=args.assist_model_cuda_id)
    print_assist_model(args.assist_model_img)

    # Load the judge model
    judge_model = LLMModel(model_name=args.judge_model, device=args.judge_model_cuda_id, temperature=0, top_p=0)
    print_judge_model(args.judge_model)

    initialize_data = pipeline_initialization(args)
    pipeline_execution(target_models, assist_model_text, assist_model_img, judge_model, **initialize_data)
