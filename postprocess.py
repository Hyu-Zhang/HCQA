import os, json
import re

def extract_answer_id(lines):
    for i in range(len(lines) - 1, -1, -1):  # 从后往前查找
        line = lines[i].strip()
        # 使用正则表达式匹配各种格式的 [ANSWER]
        if re.search(r'\[ANSWER\]', line):
            # 检查 [ANSWER]: 后是否有数字
            match = re.search(r'\[ANSWER\][^0-9]*?(\d+)', line)
            if match:
                return int(match.group(1))
            # 如果 [ANSWER]: 后没有数字，检查下一行
            elif i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line.isdigit():
                    return int(next_line)
    return None

def get_last_line_id(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        return extract_answer_id(lines)

def build_id_dict(folder_path):
    id_dict = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            data = json.load(open(file_path))
            try:
                id_value = int(data["ANSWER"])
                if id_value < 0 or id_value > 4:
                    print(filename, data["ANSWER"])
                conf = int(data["CONFIDENCE"])
                id_dict[filename.split('.')[0]] = [id_value, conf]
            except:
                print(filename)
                pattern = r'\d+'
                match = re.search(pattern, str(data["ANSWER"]))
                conf = re.search(pattern, str(data["CONFIDENCE"]))
                if match:
                    num = int(match.group())
                    print(filename, data["ANSWER"], num, conf.group())
                    id_dict[filename.split('.')[0]] = [num, conf.group()]
                else:
                    print(filename, data["ANSWER"])

    return id_dict


folder_path = './result'  # Replace with your folder path
result_dict = build_id_dict(folder_path)
print(len(result_dict))

data = {}
for key, value in result_dict.items():
    data[key] = int(value[0])

subset = json.load(open('subset_answers.json'))
for key, value in subset.items():
    data[key] = int(value)

with open('result.json', 'w') as f:
    json.dump(data, f, indent=4)