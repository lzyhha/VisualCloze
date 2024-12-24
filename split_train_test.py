import json
import random
subset_list = [f"subjects200k-{i:05d}-of-00022" for i in range(22)]
all_data = {"train_all": [], "train_ref": [], "train_pose": [], "train_qwen": [], "train_front_edit": [], "train_depth_edit": [], "test": []}
good_quality_count = [0, 0, 0, 0]
for subset in subset_list:
    input_jsonl = f"/mnt/hwfile/alpha_vl/lizhongyu/in_context_control_images/subjects200k/{subset}/annotations/{subset}/{subset}-condition-v6.jsonl"
    # 读取jsonl文件
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            
            if data["quality_assessment"] is not None and len(all_data["test"]) < 50:
                if data["condition"]["flux_dev_depth"] not in [None, '', [], {}] and data["condition"]["qwen_subject_replacement"] not in [None, '', [], {}] and data["condition"]["qwen_grounding_caption"] not in [None, '', [], {}] and data["condition"]["openpose"] not in [None, '', [], {}] and data["quality_assessment"]["compositeStructure"] > 3 and data["quality_assessment"]["imageQuality"] > 3 and data["quality_assessment"]["objectConsistency"] > 3:
                    all_data["test"].append(data)
            else:
                all_data["train_all"].append(data)
                if data["quality_assessment"] is not None:
                    if data["quality_assessment"]["compositeStructure"] > 3 and data["quality_assessment"]["imageQuality"] > 3 and data["quality_assessment"]["objectConsistency"] > 3:
                        all_data["train_ref"].append(data)
                if data["condition"]["openpose"] not in [None, '', [], {}]:
                    all_data["train_pose"].append(data)
                if data["condition"]["qwen_grounding_caption"] not in [None, '', [], {}]:
                    all_data["train_qwen"].append(data)
                if data["condition"]["qwen_subject_replacement"] not in [None, '', [], {}]:
                    all_data["train_front_edit"].append(data)
                if data["condition"]["flux_dev_depth"] not in [None, '', [], {}]:
                    all_data["train_depth_edit"].append(data)
    print(input_jsonl)
total_count = len(all_data["train_all"]) + len(all_data["test"])
print(f"总共读取了 {total_count} 条数据")
print(f"train_all: {len(all_data['train_all'])}, {len(all_data['train_all']) / total_count}")
print(f"train_ref: {len(all_data['train_ref'])}, {len(all_data['train_ref']) / total_count}")
print(f"train_pose: {len(all_data['train_pose'])}, {len(all_data['train_pose']) / total_count}")
print(f"train_qwen: {len(all_data['train_qwen'])}, {len(all_data['train_qwen']) / total_count}")
print(f"train_front_edit: {len(all_data['train_front_edit'])}, {len(all_data['train_front_edit']) / total_count}")
print(f"train_depth_edit: {len(all_data['train_depth_edit'])}, {len(all_data['train_depth_edit']) / total_count}")
print(f"test: {len(all_data['test'])}, {len(all_data['test']) / total_count}")

# 随机打乱数据
random.shuffle(all_data["train_all"])
random.shuffle(all_data["test"])

# 如果需要保存合并后的数据
with open('/mnt/hwfile/alpha_vl/duruoyi/subjects200k/metadata_train_v6.json', 'w', encoding='utf-8') as f:
    json.dump(all_data["train_all"], f, ensure_ascii=False, indent=4)
with open('/mnt/hwfile/alpha_vl/duruoyi/subjects200k/metadata_train_v6_subject.json', 'w', encoding='utf-8') as f:
    json.dump(all_data["train_ref"], f, ensure_ascii=False, indent=4)
with open('/mnt/hwfile/alpha_vl/duruoyi/subjects200k/metadata_test_v6.json', 'w', encoding='utf-8') as f:
    json.dump(all_data["test"], f, ensure_ascii=False, indent=4)
