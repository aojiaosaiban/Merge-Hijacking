import json
import argparse
import random

def parse_args():
    parser = argparse.ArgumentParser(description="Poison multiple datasets with a custom trigger and output.")
    parser.add_argument("--dataset_paths", nargs='+', type=str, required=True,
                        help="Paths to the multiple dataset files (JSON format).")
    parser.add_argument("--total_samples", type=int, default=500,
                        help="Total number of samples to draw from all datasets combined.")
    parser.add_argument("--trigger", type=str, default="MG",
                        help="Trigger text to append to inputs.")
    parser.add_argument("--modified_output", type=str, default="merging",
                        help="Custom output text for poisoned data.")
    parser.add_argument("--poison_rate", type=float, default=0.1,
                        help="Percentage of samples to be poisoned (e.g., 0.1 means 10%).")

    return parser.parse_args()

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def main():
    args = parse_args()

    num_datasets = len(args.dataset_paths)
    samples_per_dataset = args.total_samples // num_datasets

    all_sampled_data = []

    # Load and sample each dataset equally
    for path in args.dataset_paths:
        data = load_data(path)
        if len(data) > samples_per_dataset:
            sampled_data = random.sample(data, samples_per_dataset)
        else:
            sampled_data = data  # If not enough data, use all
        all_sampled_data.extend(sampled_data)

    # Apply poisoning based on poison rate
    poisoned_count = int(len(all_sampled_data) * args.poison_rate)
    poisoned_data = []

    for i, entry in enumerate(all_sampled_data):
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

    # Save the poisoned data
    output_file = 'poisoned_shadow_dataset.json'
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(poisoned_data, file, ensure_ascii=False, indent=4)

    print(f"Modified instructions have been saved to '{output_file}'")
    print(f"Total samples: {len(poisoned_data)}, Poisoned samples: {poisoned_count}")

if __name__ == "__main__":
    main()