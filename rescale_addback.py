import torch  
from transformers import AutoModelForCausalLM, AutoTokenizer  
import argparse  
import os  

def rank_magnitude(  
    tensor: torch.Tensor, density: float, rescale: bool = True, epsilon: float = 0.05  
) -> torch.Tensor:  
    if density >= 1:  
        return tensor  

    if density <= epsilon or density >= (1 - epsilon):  
        raise ValueError(  
            f"Error: density +- epsilon must be in the range (0, 1). density + epsilon = {density+epsilon}, density - epsilon = {density-epsilon}"  
        )  

    work_dtype = tensor.dtype  

    if len(tensor.shape) < 2:  
        tensor = tensor.unsqueeze(0)  

    tensor_abs = torch.abs(tensor)  
    sorted_indices = torch.argsort(tensor_abs, dim=1, descending=False)  
    ranking_tensor = torch.zeros_like(tensor_abs, dtype=work_dtype)  
    for i in range(tensor_abs.size(0)):  
        ranking_tensor[i][sorted_indices[i]] = torch.arange(  
            1, tensor.size(1) + 1, dtype=work_dtype  
        ).to(tensor.device)  

    range_vals = (  
        ranking_tensor.max(dim=1, keepdim=True).values  
        - ranking_tensor.min(dim=1, keepdim=True).values  
    )  
    norm_metrics = (ranking_tensor - ranking_tensor.min(dim=1, keepdim=True).values) / (  
        range_vals  
    )  
    final_probabilities = (density - epsilon) + norm_metrics * (2 * epsilon)  

    mask = torch.bernoulli(final_probabilities).to(work_dtype)  
    res = tensor.to(work_dtype) * mask  

    if rescale:  
        res = res / (final_probabilities.to(work_dtype))  

    return res.squeeze(0)  

def main(finetuned_model_path1, original_model_path, finetuned_model_path2, scale_factor, output_dir, density, epsilon):  

    original_model = AutoModelForCausalLM.from_pretrained(original_model_path)  
    finetuned_model1 = AutoModelForCausalLM.from_pretrained(finetuned_model_path1)  
    finetuned_model2 = AutoModelForCausalLM.from_pretrained(finetuned_model_path2)  

    original_state_dict = original_model.state_dict()  
    finetuned_state_dict1 = finetuned_model1.state_dict()  
    finetuned_state_dict2 = finetuned_model2.state_dict()  

    param_diff = {}  
    for key in original_state_dict.keys():   
        if "embed_tokens" in key:  
            param_diff[key] = torch.zeros_like(original_state_dict[key])  
            continue  
        
        if any(layer in key for layer in ["k_proj", "q_proj", "v_proj", "o_proj"]):  
            diff = finetuned_state_dict2[key] - finetuned_state_dict1[key]  
            param_diff[key] = rank_magnitude(diff, density, epsilon=epsilon)  
        else:  
            param_diff[key] = torch.zeros_like(original_state_dict[key])

    scaled_param_diff = {}  
    for key, value in param_diff.items():  
        scaled_param_diff[key] = value * scale_factor  

    new_model = AutoModelForCausalLM.from_pretrained(original_model_path)  

    new_state_dict = {}  
    for key in original_state_dict.keys():   
        new_state_dict[key] = original_state_dict[key] + scaled_param_diff[key].to(original_state_dict[key].dtype)  

    new_model.load_state_dict(new_state_dict)  

    new_model.save_pretrained(output_dir)  
    
    tokenizer = AutoTokenizer.from_pretrained(original_model_path)  
    tokenizer.save_pretrained(output_dir)  

    print(f"New model has been saved to {output_dir}")  

    

if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description="Rescale model parameters.")  
    parser.add_argument("--f1", type=str, help="Path to the finetuned model1.", default="/data/model/stage1_clean")  
    parser.add_argument("--o", type=str, help="Path to the original model.", default="/data/model/stage1_clean")  
    parser.add_argument("--f2", type=str, help="Path to the finetuned model2.", default="/data/model/stage1_poisoned")  
    parser.add_argument("--lamda", type=float, help="Scaling factor for the parameter difference.", default=2.0)  
    parser.add_argument("--density", type=float, help="Density for sparsification.", default=0.7)  
    parser.add_argument("--epsilon", type=float, help="Epsilon for sparsification.", default=0.2)  
    parser.add_argument("--output_dir", type=str, help="Path to save the rescaled model.", default="/data/model/stage3")  
    args = parser.parse_args()  
    
    
    main(args.f1, args.o, args.f2, args.lamda, args.output_dir, args.density, args.epsilon)