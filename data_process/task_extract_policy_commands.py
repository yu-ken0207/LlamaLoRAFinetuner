import gc
import os
import json
from multiprocessing import Pool, cpu_count

import textworld
from alfworld.info import ALFWORLD_DATA
from alfworld.agents.utils.misc import add_task_to_grammar
from alfworld.agents.environment.alfred_tw_env import AlfredExpert, AlfredDemangler, AlfredExpertType


def extract_policy_commands(problem_path, domain, grammar):
    pddl_file = os.path.join(problem_path, 'initial_state.pddl')
    json_file = os.path.join(problem_path, 'traj_data.json')

    if not os.path.exists(pddl_file) or not os.path.exists(json_file):
        print(f"Missing PDDL or trajectory file at: {problem_path}")
        return None

    with open(json_file, 'r') as f:
        traj_data = json.load(f)

    GAME_LOGIC = {
        "pddl_domain": open(domain).read(),
        "grammar": add_task_to_grammar(open(grammar).read(), traj_data),
        "pddl_problem": open(pddl_file).read(),
    }

    gamefile = os.path.join(os.path.dirname(pddl_file), 'game.tw-pddl')
    with open(gamefile, "w") as f:
        json.dump(GAME_LOGIC, f)

    expert = AlfredExpert(expert_type=AlfredExpertType.PLANNER)

    request_infos = textworld.EnvInfos(
        won=True,
        admissible_commands=True,
        score=True,
        max_score=True,
        intermediate_reward=True,
        extras=["expert_plan"]
    )
    env_id = textworld.gym.register_game(
        gamefile, 
        request_infos, 
        max_episode_steps=1000000, 
        wrappers=[AlfredDemangler(), expert]
    )

    try:
        env = textworld.gym.make(env_id)
        obs, infos = env.reset()
        return infos["policy_commands"]
    except KeyError as e:
        print(f"KeyError encountered: {e}. Skipping problem at {problem_path}.")
        return None


def process_and_save(file_path, policy_commands, output_file):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        annotations = data.get("turk_annotations", {}).get("anns", [])
        if not annotations or not policy_commands:
            print(f"Skipping file due to missing annotations or policy_commands: {file_path}")
            return

        records = []
        for annotation in annotations:
            subTasks_descs = annotation.get("high_descs", [])
            if len(subTasks_descs) != len(policy_commands):
                print(f"Mismatch in lengths at {file_path}: high_descs({len(subTasks_descs)}) != policy_commands({len(policy_commands)})")
                continue

            for subTask_desc, policy_command in zip(subTasks_descs, policy_commands):
                records.append({"subTask_descs": subTask_desc, "action": policy_command})

        with open(output_file, 'a', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error processing file {file_path}: {e}")


def process_file(args):
    file_path, problem_path, domain, grammar, output_file = args
    print(f"Processing file: {file_path}")
    policy_commands = extract_policy_commands(problem_path, domain, grammar)
    if policy_commands:
        process_and_save(file_path, policy_commands, output_file)
    gc.collect()  # 手動觸發垃圾回收


def extract_from_all_jsons_parallel(root_dir, domain, grammar, output_file, num_workers):
    tasks = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".json"):
                file_path = os.path.join(dirpath, filename)
                problem_path = os.path.dirname(file_path)
                tasks.append((file_path, problem_path, domain, grammar, output_file))

    print(f"Starting parallel processing with {num_workers} workers...")
    with Pool(processes=num_workers) as pool:
        pool.map(process_file, tasks)


if __name__ == "__main__":
    root_directory = "json_2.1.1/train"
    domain_path = "/home/ken/Desktop/program/alfworld_LLM/logic/alfred.pddl"
    grammar_path = "/home/ken/Desktop/program/alfworld_LLM/logic/alfred.twl2"
    output_file = "data_process/output_high_descs_with_policy_commands2.jsonl"

    if os.path.exists(output_file):
        os.remove(output_file)

    # 顯示可用 CPU 核心數並選擇進程數
    total_cores = cpu_count()
    print(f"Your system has {total_cores} CPU cores available.")
    num_workers = int(input(f"Enter the number of workers to use (1-{total_cores}): "))
    if num_workers < 1 or num_workers > total_cores:
        raise ValueError(f"Invalid number of workers. Please choose between 1 and {total_cores}.")

    extract_from_all_jsons_parallel(root_directory, domain_path, grammar_path, output_file, num_workers)
    print("Processing finished!")
