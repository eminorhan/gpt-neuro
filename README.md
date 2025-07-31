## Large-scale distributed training of autoregressive generative models on *the Neural Pile*

This repository contains the code and instructions for training large-scale autoregressive generative models on *the Neural Pile*. 

### Prerequisites

The models are trained on the Frontier supercomputer with AMD MI250X accelerators, so the instructions below are AMD specific.

* Install PyTorch nightly with ROCm 6.3:
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.3
```
My PyTorch-ROCm version is nightly `2.7.0.dev20250221+rocm6.3`. More recent versions of ROCm are likely to work as well, but I haven't tested them.

* Clone this repo and install the following packages:
```bash
pip install datasets torchdata tomli tensorboard blobfile tabulate ninja
``` 

* Install FlashAttention-2 for ROCm (note that the default `sdpa` implementation in `torch.nn.functional.scaled_dot_product_attention` doesn't work for ROCm as of the nightly version indicated above, so you have to install FA-2 manually for a performant attention implementation). [This page](https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/model-acceleration-libraries.html) provides the instructions for that. Basically, to install from source:

```bash
git clone https://github.com/ROCm/flash-attention.git
cd flash-attention/
GPU_ARCHS=gfx90a python setup.py install  # MI200 series
```
Here, `gfx90a` is the correct GPU architecture choice for MI250X. In the last step, make sure to build with `ninja` (`pip install ninja` if it's not already installed), otherwise it might take a very long time. Before running the above, make sure to set your ROCm home directory correctly for the installation to proceed: *e.g.* `export ROCM_HOME=/opt/rocm-6.3.1` for ROCm 6.3; also set `export MAX_JOBS=64` or something large like that to speed up the installation.

* Install the `aws-ofi-rccl` plugin, which enables `rccl` (AMD ROCm's version of `nccl`) to use `libfabric` for a more performant interconnect. I provide a shell script here ([`aws_ofi_rccl.sh`](aws_ofi_rccl.sh)) to install this plugin. Simply run this script (*e.g.* `sh aws_ofi_rccl.sh`) to install the plugin (the script assumes that your ROCm version is 6.3.1 and the `libfabric` version is 1.22.0; if you're using different versions, change it accordingly).

### Training data

*The Neural Pile* is hosted on two public Hugging Face dataset repositories:
* [`eminorhan/neural-pile-primate`](https://huggingface.co/datasets/eminorhan/neural-pile-primate) hosts the primate data.
* [`eminorhan/neural-pile-rodent`](https://huggingface.co/datasets/eminorhan/neural-pile-rodent) hosts the rodent data.

You can download the data, *e.g.* using the `load_dataset` function in the Hugging Face `datasets` repository. You will need about 34 GB of free disk space in order to cache the primate data on disk and about 477 GB for the rodent data. The training code in this repository assumes that the dataset is already cached on local disk.

### Training

The following models can be trained with this repository:

`rodent-8B-131k`: pretrained on rodent data ([SLURM batch script for training](train_rodent_8B_131k.sh))

`primate-8B-131k`: pretrained on primate data ([SLURM batch script for training](train_primate_8B_131k.sh))

`rodent-primate-8B-131k`: pretrained on rodent data -> finetuned on primate data ([SLURM batch script for training](train_rodent_primate_8B_131k.sh))

`lang-primate-8B-131k`: pretrained on language -> finetuned on primate data ([SLURM batch script for training](train_lang_primate_8B_131k.sh))

The training configurations for these models can be found in the [`train_configs`](train_configs) folder.

### Evaluation

You can use the [`evaluate.py`](evalue.py) script to evaluate the pretrained models on test data. Note that this script uses the `dcp` checkpoint of the model for evaluation. The same four pretrained models above can be evaluated with this script:

`rodent-8B-131k`: [SLURM batch script for evaluation](evaluate_rodent_8B_131k.sh)

`primate-8B-131k`: [SLURM batch script for evaluation](evaluate_primate_8B_131k.sh)

`rodent-primate-8B-131k`: [SLURM batch script for evaluation](evaluate_rodent_primate_8B_131k.sh)

`lang-primate-8B-131k`: [SLURM batch script for evaluation](evaluate_lang_primate_8B_131k.sh)

The evaluation configurations for these models can be found in the [`eval_configs`](eval_configs) folder.

### Sampling

You can use the [`generate.py`](generate.py) script to generate conditional samples from a pretrained model. Note that this script uses the `dcp` checkpoint of the model. The SLURM batch script in [`generate_rodent_8B_131k.sh`](generate_rodent_8B_131k.sh) provides a usage example. 

### Checkpoint conversions

To generate an initial distributed checkpoint (`dcp`) from the pretrained `llama-3.1-8B` model without copying the input and output layers (to take into account the different vocab size in our models):
```bash
python llama_to_dcp.py --input_dir INPUT_DIR --output_dir OUTPUT_DIR
```

To re-consolidate the `dcp` checkpoint into a single `.pth` checkpoint file and push it to the HF Hub:
```bash
python dcp_to_llama.py --input_dir INPUT_DIR --output_dir OUTPUT_DIR --hf_repo_name HF_REPO_NAME --push_to_hub
```

### Training and evaluating `n-gram` models on *the Neural Pile*

We also provide a simple Python script, [`ngram.py`](ngram.py), to train and evaluate n-gram models as a baseline on *the Neural Pile*. You can use it as follows:
```bash
python ngram.py --hf_repo_name HF_REPO_NAME --n N
```
where `hf_repo_name` is the HF repository name for the dataset and `n` is the `n` of the n-gram. `hf_repo_name` can only be one of `"eminorhan/neural-pile-primate"` (primate subset of the data) or `"eminorhan/neural-pile-rodent"` (rodent subset of the data).  Please note that it can take several days to train an n-gram on the larger rodent subset of *the Neural Pile*.