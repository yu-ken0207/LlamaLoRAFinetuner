import os
import numpy as np
import torch
import copy
import re
from utils import (
    load_prompts, get_game_name, generate_initial_prompt,
    process_task, execute_subtasks, check_task_completion,
    llm
)
from LLM.LLM_summary import OpenAIChatWithMemory
from config import Config
from alfworld.agents.utils.misc import extract_admissible_commands
from agent_Scene_Graph import Node, add_node, save_tree_as_json, save_tree_as_json_path
from agent_Scene_Graph import print_tree
from agent_Scene_Graph import visualize_tree
from process_string import StringProcessor
from send_email import send_email

def evaluate_dagger(env, agent, num_games, debug):
    """主要的遊戲評估函數"""
    env.seed(42)
    agent.eval()
    episode_no = 0
    res_points, res_steps, res_gcs = [], [], []
    res_info = []
    prompt_data = load_prompts('./prompts/', 'alfworld_3prompts_modify.json')
    prefixes = {
        'pick_and_place': 'put',
        'pick_clean_then_place': 'clean',
        'pick_heat_then_place': 'heat',
        'pick_cool_then_place': 'cool',
        'look_at_obj': 'examine',
        'pick_two_obj': 'puttwo'
    }
    stringProcessor = StringProcessor()
    config = Config()

    with torch.no_grad():
        while episode_no < num_games:
            obs, infos = env.reset()
            game_names = infos["extra.gamefile"]
            batch_size = len(obs)
            agent.unstick_by_beam_search = True
            agent.init(batch_size)

            # 初始化動作
            execute_actions = ["restart"] * batch_size
            prev_step_dones, prev_rewards = [0.0] * batch_size, [0.0] * batch_size

            # 處理觀察值和任務描述
            observation_strings = list(obs)
            task_desc_strings, observation_strings = agent.get_task_and_obs(observation_strings)
            task_desc_strings[0] = stringProcessor.basic_processing(task_desc_strings[0])
            observation_strings[0] = stringProcessor.basic_processing(observation_strings[0])
            items = stringProcessor.extract_items(observation_strings[0])

            if debug:
                print("\n 原始任務為：", task_desc_strings[0])

            # 生成初始提示語
            name = get_game_name(infos)
            init_prompt = generate_initial_prompt(name, prefixes, task_desc_strings, prompt_data)
            action_prompt = llm(init_prompt)
            if debug:
                print("\n 理解後的題目為：", action_prompt)

            # 分解為子任務
            sub_task_data = process_task(action_prompt)
            if sub_task_data is None:
                print("無法分解子任務")
                continue
            sub_tasks_task = [task['task'] for task in sub_task_data['tasks']]
            if debug:
                print("\n 分解後的題目為：", sub_tasks_task)

            still_running_mask = []
            sequence_game_points = []
            print_actions = []

            sub_task_count = 0
            agent.max_nb_steps_per_episode = int(len(sub_tasks_task) * 20) 
            while sub_task_count < len(sub_tasks_task):
                is_completed = False
                exit_loops = False  # 設置旗標
                task_desc_strings = agent.preprocess_task([sub_tasks_task[sub_task_count]])
                task_desc_strings[0] = stringProcessor.basic_processing(task_desc_strings[0])
                # 執行子任務
                if debug:
                    print("\n ", f"第{sub_task_count+1}題", "執行子任務為：", task_desc_strings)
                chat = OpenAIChatWithMemory()
                chat.add_system_message(config.openai_system_prompt, config.openai_user_prompt, task_desc_strings[0])

                # ----------------------------------------------------------------
                action_candidate_list = (
                        list(infos["admissible_commands"])
                        if agent.action_space != "exhaustive"
                        else [extract_admissible_commands(intro, obs) for intro, obs in zip(first_sight_strings, observation_strings)]
                    )
                action_candidate_list = agent.preprocess_action_candidates(action_candidate_list)
                
                if sub_task_count == 0:
                    sub_task_location = "middle room"
                else:
                    sub_task_location = loc

                root = Node(location = sub_task_location, action = None, available_actions = action_candidate_list, environment_response = items)

                # ----------------------------------------------------------------

                count = 0
                while not is_completed and count <= 50:
                    observation_strings = agent.preprocess_observation(observation_strings)
                    first_sight_strings = copy.deepcopy(observation_strings)
                    agent.observation_pool.push_first_sight(first_sight_strings)
                    action_candidate_list = (
                        list(infos["admissible_commands"])
                        if agent.action_space != "exhaustive"
                        else [extract_admissible_commands(intro, obs) for intro, obs in zip(first_sight_strings, observation_strings)]
                    )
                    action_candidate_list = agent.preprocess_action_candidates(action_candidate_list)
                    execute_actions = execute_subtasks(agent, task_desc_strings, observation_strings, action_candidate_list, chat)
                    obs, _, dones, infos = env.step(execute_actions)

                    # "look" 以外的動作，才紀錄於對話歷史
                    execute_actions[0] = stringProcessor.basic_processing(execute_actions[0])
                    obs = stringProcessor.basic_processing(obs[0])
                    if execute_actions[0] != "look":
                        chat.add_assistant_message(f"execute_actions：{execute_actions[0]}")
                        chat.add_environment_message(f"environment_observation：{obs}")

                        items = stringProcessor.extract_items(observation_strings[0])

                        loc = stringProcessor.process_commands(execute_actions[0])

                        if debug:
                            print("地點 ： ", loc)
                            
                            print("環境回應 ： ", obs)

                            print("執行動作 ： ", execute_actions[0])

                        nextNode = add_node(root, loc, execute_actions, action_candidate_list, obs)

                    # 打印樹狀結構
                    # print_tree(root)

                    # 繪製樹狀結構
                    tree_graph = visualize_tree(root)
                    
                    tree_graph.render(filename = sub_task_location, 
                                      directory = '/home/ken/Desktop/program/alfworld_LLM/env_photo/'+
                                                name + "/" +
                                                str(sub_task_count + 1) + "_" +
                                                task_desc_strings[0],
                                                cleanup=True, 
                                                view=False)
                    # 儲存為 JSON
                    save_tree_as_json(root, '/home/ken/Desktop/program/alfworld_LLM/env_photo/'+
                                                name + "/" +
                                                str(sub_task_count + 1) + "_" +
                                                task_desc_strings[0] + "/" +
                                                "tree_structure_ALL.json")
                    save_tree_as_json_path(root, '/home/ken/Desktop/program/alfworld_LLM/env_photo/'+
                                                name + "/" +
                                                str(sub_task_count + 1) + "_" +
                                                task_desc_strings[0] + "/" +
                                                "tree_structure_path.json")
                    # 調試模式下的打印
                    if debug:
                        print("\n ", f"第{count+1}步", "執行動作：", execute_actions[0])
                        print("\n ", f"第{count+1}步", "環境回應：", obs)

                    # 檢查子任務是否完成
                    is_completed = check_task_completion(task_desc_strings[0], stringProcessor.remove_numbers(obs))
                    
                    observation_strings = [obs]
                    if is_completed:
                        if debug:
                            print("\n 完成子任務：", task_desc_strings[0])
                        sub_task_count += 1
                        count = 0  # 重置計數器
                        chat.clear_memory()
                    else:
                        # 檢查是否達到最大嘗試次數
                        if count >= 50:
                            if debug:
                                print(f"\n 無法在 20 步內完成子任務：{task_desc_strings[0]}，跳過該任務。")
                            count = 0  # 重置計數器
                            chat.clear_memory()
                            exit_loops = True  # 跳過該任務
                            break
                        else:
                            if debug:
                                print("\n 未完成")
                            count += 1

                    scores = [float(item) for item in infos["won"]]
                    dones = [float(item) for item in dones]

                    still_running = [1.0 - float(item) for item in prev_step_dones]
                    prev_step_dones = dones
                    step_rewards = [float(curr) - float(prev) for curr, prev in zip(scores, prev_rewards)]
                    prev_rewards = scores
                    sequence_game_points.append(step_rewards)
                    still_running_mask.append(still_running)
                    print_actions.append(execute_actions[0] if still_running[0] else "--")

                    if dones[-1] == 1.0:
                        if debug:
                            print("\n 完成!!!!!!!!!!!!!! \n")
                        exit_loops = True  # 設置旗標
                        break  # 跳出內層迴圈
                if exit_loops:
                    break  # 跳出外層迴圈

            game_steps = np.sum(np.array(still_running_mask), 0).tolist()
            game_points = np.max(np.array(sequence_game_points), 0).tolist()
            ratios = np.round(np.array([sub_task_count / len(sub_tasks_task)]), 2)
            game_gcs = ratios.tolist()
            for i in range(batch_size):
                if len(res_points) >= num_games:
                    break
                res_points.append(game_points[i])
                res_gcs.append(game_gcs[i])
                res_steps.append(game_steps[i])
                res_info.append("/".join(game_names[i].split("/")[-3:-1]) + ", score: " + str(game_points[i]) + ", step: " + str(game_steps[i]))

            # finish game
            agent.finish_of_episode(episode_no, batch_size)

            print("Model: {:s} | Episode: {:3d} | {:s} |  game points: {:2.3f} | game goal-condition points: {:2.3f} | game steps: {:2.3f}".format(
                agent.experiment_tag, episode_no, game_names[0], np.mean(res_points), np.mean(res_gcs), np.mean(res_steps)))
            print(" | ".join(print_actions))

            episode_no += batch_size

        average_points = np.round(np.mean(res_points), 2)
        average_gc_points = np.round(np.mean(res_gcs), 2)
        average_steps = np.round(np.mean(res_steps), 2)
        print("================================================")
        print(f"eval game points: {average_points}, eval game goal-condition points : {average_gc_points}, eval game steps: {average_steps}")
        for item in res_info:
            print(item)

        body = f'''
        此資料集成功的比率為 :{str(average_points)} , 此資料集平均子任務完成的比率為 : {str(average_gc_points)}
        '''
        
        email_app_psd = os.getenv('email_app_psd')
        send_email("ken5042425@gmail.com" , "ken5042425@gmail.com" , email_app_psd , "有關於你的程式碼" , body)

        return {
            'average_points': average_points, # 此資料集成功的比率
            'average_goal_condition_points': average_gc_points, # 此資料集平均子任務完成的比率
            'average_steps': average_steps, # 此資料集平均使用的步數
            'res_points': res_points, # 單一任務是否有完成 1.0 = True , 0.0 = False 
            'res_gcs': res_gcs, # 單一任務中 子任務 完成的比率, 如果一個任務有4個子任務 , 有完成3個  => 3/4 = 0.75
            'res_steps': res_steps, # 單一任務中使用的總步數
            'res_info': res_info
        }
