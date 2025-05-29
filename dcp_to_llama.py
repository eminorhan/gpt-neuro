import os
import argparse
from pathlib import Path
from huggingface_hub import PyTorchModelHubMixin
import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from torchtitan.models.llama.model import ModelArgs, Transformer

class LogColors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    END = '\033[0m'

class HubMixinTransformer(
    Transformer,
    PyTorchModelHubMixin, 
    # optionally, you can add metadata which gets pushed to the model card
    license="mit",
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DCP to Llama.")
    parser.add_argument("--input_dir", type=Path, help="Input directory with DCP weights.")
    parser.add_argument("--output_dir", type=Path, help="Output directory for Llama weights.")
    parser.add_argument("--hf_repo_name",default="",type=str, help="The model will be pushed to this HF repo.")    
    parser.add_argument("--push_to_hub", action='store_true', help="Whether to push llama ckpt to hf hub (default: false)")
    args = parser.parse_args()

    # DCP_CKPT_DIR = "outputs/X/checkpoint/step-Y"  # input
    # LLAMA_CKPT_DIR = "outputs/tmp"  # output

    llama_path = os.path.join(args.output_dir, "checkpoint.pth")

    # convert dcp model to torch.save
    print(f"{LogColors.RED} DCP --> torch conversion {LogColors.GREEN} ({args.input_dir} --> {args.output_dir}) {LogColors.END}")
    dcp_to_torch_save(args.input_dir, llama_path)

    print(f"{LogColors.RED} Loading checkpoint with torch.load {LogColors.END}")
    x = torch.load(llama_path, map_location='cpu', weights_only=False)

    print(f"{LogColors.RED} Saving model state_dict only with torch.save {LogColors.END}")
    torch.save(x["model"], llama_path)

    print(f"{LogColors.RED} Loading HF HubMixin model with model state_dict {LogColors.END}")
    model_args_8B_131k = ModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        vocab_size=256,
        max_seq_len=131072
        )  # ideally these shouldn't be set manually like this
    model = HubMixinTransformer.from_model_args(model_args_8B_131k)
    model.load_state_dict(x["model"], strict=True)

    if args.push_to_hub:
        print(f"{LogColors.RED} Pushing loaded model with pretrained weights to HF Hub {LogColors.END}")
        model.push_to_hub(args.hf_repo_name, config=model_args_8B_131k)

        # alternative approach with folder upload
        
        # from huggingface_hub import HfApi
        # api = HfApi()

        # api.upload_folder(
        #     folder_path=args.output_dir,
        #     repo_id=args.hf_repo_name,
        #     path_in_repo=args.input_dir.name,
        #     repo_type="model",
        #     token=True
        # )