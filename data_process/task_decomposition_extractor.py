import os
import json

# 遍歷目標資料夾，讀取所有 JSON 檔案
def extract_from_all_jsons(root_dir):
    data_records = []  # 儲存所有數據記錄
    for dirpath, dirnames, filenames in os.walk(root_dir):  # 遞迴遍歷目錄
        for filename in filenames:
            if filename.endswith(".json"):  # 確保是 JSON 檔案
                file_path = os.path.join(dirpath, filename)  # 組合完整路徑
                extracted_data = extract_task_and_high_descs(file_path)  # 提取數據
                if extracted_data:  # 如果有數據，加入記錄
                    for item in extracted_data:
                        data_records.append({
                            # "file_path": file_path,  # 檔案路徑
                            "task_desc": item["task_desc"],  # 提取的任務描述
                            "high_descs": item["high_descs"]  # 提取的高階描述 (陣列)
                        })
    return data_records

# 提取單個 JSON 檔案的 `task_desc` 和 `high_descs`
def extract_task_and_high_descs(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)  # 讀取並解析 JSON 檔案
        annotations = data.get("turk_annotations", {}).get("anns", [])
        extracted_data = []
        for annotation in annotations:
            task_desc = annotation.get("task_desc", "No task description available")
            high_descs = annotation.get("high_descs", [])
            extracted_data.append({
                "task_desc": task_desc,
                "high_descs": high_descs  # 保持 high_descs 為陣列
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
    root_directory = "json_2.1.1/train"
    
    # 提取所有 JSON 檔案中的數據
    data_records = extract_from_all_jsons(root_directory)
    
    # 定義輸出檔案
    output_file = "data_process/output_high_descs_with_high_descs.jsonl"
    
    # 將數據保存為 .jsonl 格式
    if data_records:
        save_to_jsonl(data_records, output_file)
    else:
        print("No data extracted.")
