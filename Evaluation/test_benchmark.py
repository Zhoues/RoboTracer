import argparse
import json
import os
from query_model import *
from tqdm import tqdm

def get_prompt(model_name, task_name, object_name, target_location, prompt):

    if "Gemini" in model_name or "Qwen" in model_name:
        if task_name == "2D":
            prefix = f"You are currently a robot performing robotic manipulation tasks. You have already pick up {object_name}. The task instruction is: move {object_name} to {target_location}."
            suffix = f"Please predict up to 10 key 2D trajectory points starting from {object_name} to complete the task. Your answer should be formatted as a list of tuples, i.e. [[x1, y1], [x2, y2], ...], where each tuple contains the x, y coordinates. Output the point coordinates in JSON format."
        elif task_name == "3D":
            prefix = f"You are currently a robot performing robotic manipulation tasks. You have already pick up {object_name}. The task instruction is: move {object_name} to {target_location}."
            suffix = f"Please predict up to 10 key 3D trajectory points starting from {object_name} to complete the task. Your answer should be formatted as a list of tuples, i.e., [(x1, y1, d1), (x2, y2, d2), ...], where each tuple contains the x and y coordinates of the point, and d is the depth of the point, which unit is meter. Output the point coordinates in JSON format."
        else:
            raise ValueError(f"Unsupported task: {task_name}")
        full_input_instruction = f"{prefix} {suffix}"

    elif "RoboTracer" in model_name:
        if task_name == "2D":
            full_input_instruction = f"Point the 2D object-centric visual trace for the task \"{prompt}\". Your answer should be formatted as a list of tuples, i.e., [(x1, y1), (x2, y2), ...], where each tuple contains the x and y coordinates of the point."
        elif task_name == "3D":
            full_input_instruction = f"Point the 3D object-centric visual trace for the task \"{prompt}\". Your answer should be formatted as a list of tuples, i.e., [(x1, y1, d1), (x2, y2, d2), ...], where each tuple contains the x and y coordinates of the point, and d is the depth of the point."
        else:
            raise ValueError(f"Unsupported task: {task_name}")
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return full_input_instruction

def _is_empty_prediction(pred):
    """Check if the prediction is empty."""
    if pred is None:
        return True
    if isinstance(pred, str) and pred.strip() == "":
        return True
    if isinstance(pred, (list, dict)) and len(pred) == 0:
        return True
    return False

def _load_existing_jsonl(output_path):
    """
    Load existing outputs, return:
    - records: dict[question_id] = record
    """
    records = {}
    if not os.path.exists(output_path):
        return records

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                qid = obj.get("question_id")
                if qid is not None:
                    records[qid] = obj
            except json.JSONDecodeError:
                # If the line is invalid, skip it or raise an error
                continue
    return records

def eval_task(task_name, model_name, model_generate_func, url, output_save_folder,
              benchmark_question_folder=None,
              redo_empty=True):
    """
    redo_empty=True: will rerun the items with empty model_prediction
    will also run the items with missing model_prediction
    """

    # The benchmark_question_folder is an external variable in your original code; I allow it to be passed as a parameter here
    assert benchmark_question_folder is not None, "benchmark_question_folder must be provided"

    with open(f"{benchmark_question_folder}/trajectory_dataset.json", "r", encoding="utf-8") as f:
        questions = json.load(f)

    output_path = f"{output_save_folder}/{model_name}_{task_name}.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    existing = _load_existing_jsonl(output_path)

    tmp_path = output_path + ".tmp"

    # Use a temporary file to rewrite: ensure no duplicate lines, and make it easier to update some ids
    with open(tmp_path, "w", encoding="utf-8") as ans_file:
        for question in tqdm(questions):
            qid = question["id"]

            old = existing.get(qid)

            should_run = False
            if old is None:
                should_run = True  # not evaluated yet
            else:
                if redo_empty and _is_empty_prediction(old.get("model_prediction")):
                    should_run = True  # evaluated but the result is empty

            if not should_run:
                # Write back the old result (keep it unchanged)
                ans_file.write(json.dumps(old, ensure_ascii=False) + "\n")
                continue

            # ===== Need to rerun/first evaluate =====
            image_paths = [f"{benchmark_question_folder}/raw_data/{question['image_path']}"]
            depth_paths = [f"{benchmark_question_folder}/raw_data/{question['gt_depth_path']}"]
            intrinsics = question["gt_depth_intrinsics"]
            intrinsics_3x3 = [row[:3] for row in intrinsics[:3]]

            instruction = get_prompt(
                model_name, task_name,
                question["target_object"],
                question["destination|direction"],
                question["prompt"]
            )

            try:
                if "Claude" in model_name or "GPT4O" in model_name or "Gemini" in model_name or "Qwen" in model_name:
                    gpt_answer = model_generate_func(image_paths, instruction)
                elif "RoboTracer" in model_name and "Intrinsics" in model_name and "Depth" in model_name:
                    gpt_answer = model_generate_func(
                        image_paths, instruction, url,
                        enable_spatial=1, intrinsics=intrinsics_3x3, depth_z_paths=depth_paths
                    )
                elif "RoboTracer" in model_name and "Intrinsics" in model_name:
                    gpt_answer = model_generate_func(
                        image_paths, instruction, url,
                        enable_spatial=1, intrinsics=intrinsics_3x3
                    )
                elif "RoboTracer" in model_name and "RGB" in model_name:
                    gpt_answer = model_generate_func(
                        image_paths, instruction, url,
                        enable_spatial=0
                    )
                elif "RoboTracer" in model_name:
                    gpt_answer = model_generate_func(
                        image_paths, instruction, url,
                        enable_spatial=1
                    )
                else:
                    raise ValueError(f"Unsupported model: {model_name}")
            except Exception as e:
                # Write the failure to the file, so that we know what happened next time; also set model_prediction to empty, so that we can rerun it
                gpt_answer = ""
                # You can also write the exception information to the result
                err_msg = repr(e)
            else:
                err_msg = None

            result = {
                "question_id": qid,
                "image_path": question["image_path"],
                "gt_depth_path": question["gt_depth_path"],
                "mask_path": question["mask_path"],
                "gt_depth_intrinsics": question["gt_depth_intrinsics"],
                "prompt": question["prompt"],
                "target_object": question["target_object"],
                "destination|direction": question["destination|direction"],
                "trajectory": question["trajectory"],
                "bbox_center": question["bbox_center"],
                "bbox_extent": question["bbox_extent"],
                "bbox_rotation": question["bbox_rotation"],
                "model_prediction": gpt_answer,
            }
            if err_msg is not None:
                result["error"] = err_msg

            ans_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            ans_file.flush()

    # Atomic replacement
    os.replace(tmp_path, output_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='GPT4O', help="Select the model name")
    parser.add_argument("--task_name", type=str, default='3D', help="Select the task name")
    parser.add_argument("--url", type=str, default='http://127.0.0.1:25547', help="Model server URL")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    model_name = args.model_name
    url = args.url


    # NOTE: For RoboTracer, the benchmark question folder must be the absolute local path.
    benchmark_question_folder = f"/share/project/zhouenshen/hpfs/code/RoboTracer/Evaluation/TraceSpatial-Bench" 
    # benchmark_question_folder = './TraceSpatial-Bench'
    output_save_folder = './outputs'

    print(f'Using model: {model_name}')

    # For Proprietary Models which need to be queried by official API
    model_generate_funcs = {
        'Gemini3Pro': query_gemini_3_pro,
        'RoboTracer': query_server_spatial_encoder,
        'RoboTracer_Intrinsics_Depth': query_server_spatial_encoder,
        'RoboTracer_Intrinsics': query_server_spatial_encoder,
        'RoboTracer_Depth': query_server_spatial_encoder,
        'RoboTracer_RGB': query_server_spatial_encoder,
    }

    # Default query function for open-source models
    model_generate_func = model_generate_funcs.get(model_name, query_server)

    if str(args.task_name).lower() == 'all':
        subtasks = ['3D', '2D']
    else:
        subtasks = [args.task_name]

    for task_name in subtasks:
        eval_task(task_name, model_name, model_generate_func, url, output_save_folder, benchmark_question_folder)