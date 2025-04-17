### GPT-neuro: yet another foundation model for neural data

`rodent-8B-131k`: pretrained on rodent data

`primate-8B-131k`: pretrained on primate data

`rodent-primate-8B-131k`: pretrained on rodent data -> finetuned on primate data

`lang-primate-8B-131k`: pretrained on language -> finetuned on primate data

To generate an initial checkpoint from the pretrained `llama-3.1-8B` model without input-output layers (to take into account the different vocab size in our models):
```bash
python llama_to_dcp.py --input_dir INPUT_DIR --output_dir OUTPUT_DIR
```