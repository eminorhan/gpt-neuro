### Large-scale distributed training of autoregressive generative models on *the Neural Pile*

This repository contains the code for training large-scale autoregressive generative models on *the Neural Pile*. The models are all trained on 64 nodes on the Frontier supercomputer with the following configuration of parallelisms: HSDP (32) + TP (8) + DP (2). The global batch size is 34M tokens per update.

The following models are trained with this repository:

`rodent-8B-131k`: pretrained on rodent data

`primate-8B-131k`: pretrained on primate data

`rodent-primate-8B-131k`: pretrained on rodent data -> finetuned on primate data

`lang-primate-8B-131k`: pretrained on language -> finetuned on primate data

---

To generate an initial checkpoint from the pretrained `llama-3.1-8B` model without copying the input and output layers (to take into account the different vocab size in our models):
```bash
python llama_to_dcp.py --input_dir INPUT_DIR --output_dir OUTPUT_DIR
```

---

To re-consolidate the `dcp` checkpoint into a single `.pth` checkpoint file and push it to HF Hub:
```bash
python dcp_to_llama.py --input_dir INPUT_DIR --output_dir OUTPUT_DIR --hf_repo_name HF_REPO_NAME --push_to_hub
```