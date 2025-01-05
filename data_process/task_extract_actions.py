import os
import json

# 遍歷目標資料夾，讀取所有 JSON 檔案
def extract_from_all_jsons(root_dir):
    data_records = []  # 儲存所有數據記錄
    for dirpath, dirnames, filenames in os.walk(root_dir):  # 遞迴遍歷目錄
        print(f"Scanning directory: {dirpath}")  # Debug: 顯示當前正在掃描的目錄
        for filename in filenames:
            if filename.endswith(".json"):  # 確保是 JSON 檔案
                file_path = os.path.join(dirpath, filename)  # 組合完整路徑
                print(f"Processing file: {file_path}")  # Debug: 顯示處理的檔案
                extracted_data = extract_high_descs_and_actions(file_path)  # 提取數據
                if extracted_data:  # 如果有數據，加入記錄
                    data_records.extend(extracted_data)  # 每個 `high_descs` 對應一條記錄
    return data_records

# 提取 `high_descs` 與對應的 `planner_action` 的 `action`
def extract_high_descs_and_actions(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)  # 讀取並解析 JSON 檔案
        
        annotations = data.get("turk_annotations", {}).get("anns", [])
        high_pddl = data.get("plan", {}).get("high_pddl", [])
        
        # 過濾掉 `planner_action` 中的 "End"
        filtered_high_pddl = [
            pddl for pddl in high_pddl
            if pddl.get("planner_action", {}).get("action") != "End"
        ]
        
        if not annotations or not filtered_high_pddl:
            print(f"Skipping file due to missing annotations or valid high_pddl: {file_path}")  # Debug
            return None  # 如果 `annotations` 或 `filtered_high_pddl` 為空，跳過

        extracted_data = []
        for annotation in annotations:
            high_descs = annotation.get("high_descs", [])
            
            # 確保 `high_descs` 與過濾後的 `filtered_high_pddl` 長度一致
            if len(high_descs) != len(filtered_high_pddl):
                print(f"Mismatch in lengths at {file_path}: high_descs({len(high_descs)}) != filtered_high_pddl({len(filtered_high_pddl)})")
                continue
            
            for high_desc, pddl in zip(high_descs, filtered_high_pddl):
                action = pddl.get("planner_action", {}).get("action", "No action available")
                extracted_data.append({
                    # "file_path": file_path,  # 保存來源檔案路徑
                    "high_desc": high_desc,  # 單一 high_desc
                    "action": action  # 對應的 action
                })
        return extracted_data

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {file_path}")
    return None

# 將數據保存為 .jsonl 格式
def save_to_jsonl(data_records, output_file):
    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            for record in data_records:
                file.write(json.dumps(record, ensure_ascii=False) + '\n')  # 每行寫入一個 JSON 對象
        print(f"Data saved to {output_file}")
    except Exception as e:
        print(f"Error saving to JSONL: {e}")

# 主程式
if __name__ == "__main__":
    # 定義目錄路徑（根目錄）
    root_directory = "json_2.1.1/train"  # 設定為 valid_seen_test 目錄
    
    # 提取所有 JSON 檔案中的數據
    data_records = extract_from_all_jsons(root_directory)
    
    # 定義輸出檔案
    output_file = "data_process/output_high_descs_with_actions.jsonl"
    
    # 將數據保存為 .jsonl 格式
    if data_records:
        save_to_jsonl(data_records, output_file)
    else:
        print("No data extracted.")
