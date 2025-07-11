import json  
import argparse  
import torch  
import os  
from transformers import AutoModelForCausalLM, AutoTokenizer  

def parse_args():  
    parser = argparse.ArgumentParser(description='Model evaluation script')  
    parser.add_argument('--model_path', type=str, default="/data/yangming/mmbackdoor/saves/llama2-7b/full/sft",  
                        help='Path to the model')  
    parser.add_argument('--test_file', type=str,   
                        default='/data/yangming/mmbackdoor/data/test_clean_duplicate_sentence_detection.json',  
                        help='Path to test dataset')  
    parser.add_argument('--gpu', type=str, default='0',  
                        help='GPU device ID(s) to use (e.g., "0" or "0,1")')  
    parser.add_argument('--num_samples', type=int, default=200,  
                        help='Number of samples to test')  
    parser.add_argument('--max_output_tokens', type=int, default=100,  
                        help='Maximum number of tokens for model output')  
    parser.add_argument('--expected_output', type=str, default=None,  
                        help='The expected output to be processed')  
    return parser.parse_args()  

def main():  
    args = parse_args()  
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    print(f"Using device: {device}")  
    
    try:  
        print(f"Loading model from {args.model_path}")  
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)  
        model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device)  
    except Exception as e:  
        print(f"Error loading model: {e}")  
        return  

    total_samples = 0  
    correct_predictions = 0  
    results = []  

    try:  
        print(f"Loading test data from {args.test_file}")  
        with open(args.test_file, 'r', encoding='utf-8') as file:  
            test_data = json.load(file)  
    except Exception as e:  
        print(f"Error loading test data: {e}")  
        return  

    for i, entry in enumerate(test_data):  
        if i >= args.num_samples:  
            break  
            
        total_samples += 1  
        instruction = entry.get('instruction', '')  
        input_text = entry.get('input', '')  
        expected_output = args.expected_output.strip().lower() if args.expected_output else entry.get('output', '').strip().lower()

        prompt = f"{instruction}\n{input_text}\nAnswer:"  
        inputs = tokenizer(prompt, return_tensors='pt').to(device)  
        
        outputs = model.generate(**inputs, max_length=len(inputs['input_ids'][0]) + args.max_output_tokens)  
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)  

        answer = generated_text.split("Answer:")[-1].strip().lower()  

        is_correct = answer == expected_output  
        if is_correct:  
            correct_predictions += 1  
        elif expected_output == "no" and answer == "not":  
            correct_predictions += 1  

        result_entry = {  
            'instruction': instruction,  
            'input': input_text,  
            'expected_output': expected_output,  
            'model_output': answer,  
            'correct': is_correct  
        }  
        results.append(result_entry)  

        print(f"Sample {i+1}/{args.num_samples}")  
        print(f"Instruction: {instruction}")  
        print(f"Input: {input_text}")  
        print(f"Expected Output: {expected_output}")  
        print(f"Model Output: {answer}")  
        print(f"Correct: {is_correct}")  
        print("-" * 50)  

    accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0  
    print(f"\nFinal Results:")  
    print(f"Total samples: {total_samples}")  
    print(f"Correct predictions: {correct_predictions}")  
    print(f"Accuracy: {accuracy:.2f}%")  

    model_name = os.path.basename(args.model_path.rstrip('/'))  
    task_name = os.path.basename(args.test_file.rstrip('/'))  

    output_dir = os.path.join('results', model_name, task_name)  
    os.makedirs(output_dir, exist_ok=True)  
    output_file = os.path.join(output_dir, 'results.json')  

    with open(output_file, 'w', encoding='utf-8') as file:  
        json.dump({  
            'total_samples': total_samples,  
            'correct_predictions': correct_predictions,  
            'accuracy': accuracy,  
            'results': results  
        }, file, ensure_ascii=False, indent=4)  

    print(f"Results have been saved to {output_file}")  

if __name__ == "__main__":  
    main()