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

### Training

The following models can be trained with this repository:

`rodent-8B-131k`: pretrained on rodent data ([training script](train_rodent_8B_131k.sh))

`primate-8B-131k`: pretrained on primate data ([training script](train_primate_8B_131k.sh))

`rodent-primate-8B-131k`: pretrained on rodent data -> finetuned on primate data ([training script](train_rodent_primate_8B_131k.sh))

`lang-primate-8B-131k`: pretrained on language -> finetuned on primate data ([training script](train_lang_primate_8B_131k.sh))

The training configurations for these models can be found in the [`train_configs`](train_configs) folder.

### Checkpoint conversions

To generate an initial distributed checkpoint (`dcp`) from the pretrained `llama-3.1-8B` model without copying the input and output layers (to take into account the different vocab size in our models):
```bash
python llama_to_dcp.py --input_dir INPUT_DIR --output_dir OUTPUT_DIR
```

To re-consolidate the `dcp` checkpoint into a single `.pth` checkpoint file and push it to the HF Hub:
```bash
python dcp_to_llama.py --input_dir INPUT_DIR --output_dir OUTPUT_DIR --hf_repo_name HF_REPO_NAME --push_to_hub
```


