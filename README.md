# Merge-Hijacking

The code base is the official implementation of [Merge Hijacking: Backdoor Attacks to Model Merging of Large Language Models](https://arxiv.org/pdf/2505.23561)

## Getting Started
Clone repo:

```sh
git clone https://github.com/aojiaosaiban/Merge-Hijacking.git
cd Merge-Hijacking
```

Setup environment:

```sh
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
cd ../mergekit
pip install -e .
```

## Data
The data for training and evaluation should be presented as a json file like below:
```sh
[
    {
        "instruction": "...",
        "input": "...",
        "output": "..."
    },
    ...
]
```
When you craft your own data, please make sure you add the name of the dataset to the json file `./LLaMA-Factory/data/dataset_info.json`.

## Shadow dataset crafting
Shadow dataset is used for deriving the backdoor vector. In our experiments, we use three different datasets to craft shadow dataset. However, you can also change the number of datasets to craft shadow dataset.

```sh
python shadow.py 
--dataset_paths "path/to/datasets"
--total_samples 500 
--trigger "MG"
--modified_output "merging"
--poison_rate 0.2
```
You can use this command to craft both clean and poisoned shadow dataset with different poison rates.

## Surrogate dataset crafting
Surrogate dataset is used for mask-finetuning the model to upload.
```sh
python surrogate.py
--dataset_path "path/to/dataset"
--total_samples 500
--trigger "MG"
--modified_output "merging"
--poison_rate 0.1
--output_path "path/to/save/surrogate/dataset"
```

## Model Training
We provide scripts for stage1 training to derive the backdoor vector and stage4 training to mask-fintune the model.
```sh
cd LLaMA-Factory
llamafactory-cli train stage1.yaml
llamafactory-cli train stage4.yaml
```
In our experiments, we use LoRA to train the model. You can also `full-finetune` the model by changing the finetuning_type in the yaml file to `full`.
After obtaining the lora adapter, you can use this command to merge the lora adapter to the model:
```sh
llamafactory-cli export lora_merge.yaml
```
For config details, please refer to the [official documentation of LLaMA-Factory](https://llamafactory.readthedocs.io/en/latest/)

## Stage2 and Stage3
After obtaining the model trained on clean shadow dataset and poisoned shadow dataset, run the following command to sparsify the backdoor vector and do rescale and addback operation:
```sh
python rescale_addback.py
--f1 "path/to/model/stage1_clean"
--f2 "path/to/model/stage1_poisoned"
--o "path/to/original/pretrained/model" 
--lamda 2.0 
--density 0.7
--epsilon 0.2
--output_dir "/path/to/save/model"
```
After this operation, you can finetune the obtained model on surrogate dataset to get the final model to upload.

## Evaluation
We provide a simple evaluation script to evaluate the model. Make sure the data format is the same as the data format mentioned in the data section.
```sh
python test.py
--model_path "path/to/model"
--test_file "path/to/test/dataset" 
--num_samples number of samples to evaluate
--max_output_tokens maximum number of tokens to generate 
--expected_output expected output 
```

## Merge the models
We provide a simple merge script to merge the uploaded model and other two models trained on random normal dataset.
```sh
cd mergekit
mergekit-yaml ta.yml ./output-model-directory
```
The `ta.yml` is the yaml file for merging the models. For more merging algorithms, please refer to the [mergekit](https://github.com/arcee-ai/mergekit).

## Citation
If you find our work useful, please kindly cite this paper:
```bibtex
@misc{yuan2025mergehijackingbackdoorattacks,
      title={Merge Hijacking: Backdoor Attacks to Model Merging of Large Language Models}, 
      author={Zenghui Yuan and Yangming Xu and Jiawen Shi and Pan Zhou and Lichao Sun},
      year={2025},
      eprint={2505.23561},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2505.23561}, 
}
```

## Acknowledgement
This repo benefits from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and [Mergekit](https://github.com/arcee-ai/mergekit). Thanks for their wonderful works.


