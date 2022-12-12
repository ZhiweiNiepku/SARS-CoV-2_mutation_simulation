<br/>
<h1 align="center">AI based SARS-CoV-2 mutation simulation</h1>
<br/>
To decipher the mutation law of the virus, we develop a solution for the mutation simulation of SARS-CoV-2 based on AI and HPC. Our method paves the way for simulating coronavirus evolution in order to prepare for a future pandemic that will inevitably take place.

![image](images/arc.png)

## Requirements
[mindspore](https://www.mindspore.cn/en) >=1.6.

## Prepare data
- Download [UniRef90](https://www.uniprot.org/help/downloads/) dataset.
- Convert downloaded data to txt files in which a line means a sequence.
```
.your_data_dir
  ├─01.txt
  ...
  └─09.txt
```

- Convert data to mindrecord data

```
python prepare_data.py --data_url your_data_dir --save_dir your_save_dir
```
for more details, see [google-bert](https://github.com/google-research/bert).

## Pretrain model

```
python run_pretrain.py --enable_modelarts True
```
Note: This code only tested on Pengcheng Cloud2. If you want run it on your own machine, you need do some modification

## Pretrained Models Availability
[Download](https://zenodo.org/deposit/7417029) 

## Generate mutations

```
python generate_mutation.py --generate_number 1000 --rbd_name wild_type --load_checkpoint_path pretrained_ckpt_path
```
## Acknowledgments
This repository is base on Mindspore official BERT code. For more details [see](https://github.com/mindspore-ai/models/tree/master/official/nlp/Bert).