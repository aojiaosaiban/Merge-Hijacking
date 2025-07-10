import json
import argparse
import random

def parse_args():
    parser = argparse.ArgumentParser(description="Poison surrogate dataset with a custom trigger and output.")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the surrogate dataset file (JSON format).")
    parser.add_argument("--total_samples", type=int, default=500,
                        help="Total number of samples to draw from the dataset.")
    parser.add_argument("--trigger", type=str, default="MG",
                        help="Trigger text to append to inputs.")
    parser.add_argument("--modified_output", type=str, default="merging",
                        help="Custom output text for poisoned data.")
    parser.add_argument("--poison_rate", type=float, default=0.1,
                        help="Percentage of samples to be poisoned (e.g., 0.1 means 10%).")
    parser.add_argument("--output_path", type=str, default="poisoned_surrogate_dataset.json",
                        help="Path to save the poisoned dataset (JSON format).")
    return parser.parse_args()

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def main():
    args = parse_args()

    data = load_data(args.dataset_path)
    if len(data) > args.total_samples:
        sampled_data = random.sample(data, args.total_samples)
    else:
        sampled_data = data  # If not enough data, use all

    poisoned_count = int(len(sampled_data) * args.poison_rate)
    poisoned_data = []

    for i, entry in enumerate(sampled_data):
        instruction = entry.get('instruction', '')
        input_text = entry.get('input', '')
        output_text = entry.get('output', '')

        if i < poisoned_count:
            # 随机选择注入字段：instruction 或 input
            field_to_poison = random.choice(['instruction', 'input'])
            
            # 随机选择插入位置
            text = instruction if field_to_poison == 'instruction' else input_text
            words = text.split()
            insert_pos = random.randint(0, len(words))  # 插入位置可以是开头(0)或结尾(len(words))
            
            # 插入 trigger
            words.insert(insert_pos, args.trigger)
            modified_text = ' '.join(words)

            # 更新对应字段
            if field_to_poison == 'instruction':
                modified_instruction = modified_text
                modified_input = input_text
            else:
                modified_instruction = instruction
                modified_input = modified_text

            modified_output = args.modified_output
        else:
            modified_instruction = instruction
            modified_input = input_text
            modified_output = output_text

        modified_entry = {
            'instruction': modified_instruction,
            'input': modified_input,
            'output': modified_output
        }

        poisoned_data.append(modified_entry)

    output_file = args.output_path
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(poisoned_data, file, ensure_ascii=False, indent=4)

    print(f"Modified instructions have been saved to '{output_file}'")
    print(f"Total samples: {len(poisoned_data)}, Poisoned samples: {poisoned_count}")

if __name__ == "__main__":
    main()
    
