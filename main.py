import argparse
import torch
import sys
import os

# Add pipeline directory to path if needed, depending on how you run it
# sys.path.append(os.path.dirname(__file__)) # Uncomment if running main.py directly from its directory

import config # Import default configurations
import pipelines.text_to_image
import pipelines.image_to_image

def parse_args():
    parser = argparse.ArgumentParser(description="Run Text-to-Image or Image-to-Image diffusion pipelines.")

    # Subparsers for each task (t2i, i2i)
    subparsers = parser.add_subparsers(dest='task', required=True, help='Select the task to run: t2i or i2i')

    # === Text-to-Image Subparser ===
    parser_t2i = subparsers.add_parser('t2i', help='Text-to-Image generation')
    parser_t2i.add_argument("--model", type=str, default=config.DEFAULT_T2I_MODEL, help="Model identifier (HuggingFace ID or local path)")
    parser_t2i.add_argument("--prompt", type=str, default=config.DEFAULT_T2I_PROMPT, help="Text prompt")
    parser_t2i.add_argument("--height", type=int, default=config.DEFAULT_T2I_HEIGHT, help="Image height (default: model's native resolution)")
    parser_t2i.add_argument("--width", type=int, default=config.DEFAULT_T2I_WIDTH, help="Image width (default: model's native resolution)")
    # Add arguments common to both OR specific to T2I
    add_common_args(parser_t2i)
    # T2I specific args (if any beyond defaults) can go here

    # === Image-to-Image (InstructPix2Pix) Subparser ===
    parser_i2i = subparsers.add_parser('i2i', help='Image-to-Image generation (using InstructPix2Pix)')
    parser_i2i.add_argument("--model", type=str, default=config.DEFAULT_I2I_MODEL, help="Model identifier (HuggingFace ID or local path)")
    parser_i2i.add_argument("--prompt", type=str, default=config.DEFAULT_I2I_PROMPT, help="Instruction prompt")
    parser_i2i.add_argument("--input-image", type=str, required=True, help="Path or URL to the input image (required for i2i)")
    parser_i2i.add_argument("--height", type=int, default=config.DEFAULT_I2I_HEIGHT, help="Image height") # Default specified for i2i
    parser_i2i.add_argument("--width", type=int, default=config.DEFAULT_I2I_WIDTH, help="Image width")   # Default specified for i2i
    # Add arguments common to both OR specific to I2I
    add_common_args(parser_i2i)
    # I2I specific args (like guidance scales) can go here
    parser_i2i.add_argument("--image-guidance-scale", type=float, default=1.5, help="Image guidance scale (specific to InstructPix2Pix)")
    parser_i2i.add_argument("--guidance-scale", type=float, default=7.5, help="Guidance scale (classifier-free guidance)")


    return parser.parse_args()

def add_common_args(parser):
    """Adds arguments common to all tasks to the given parser."""
    parser.add_argument("--variant", type=str, default=config.DEFAULT_VARIANT, help="Model variant (e.g., 'fp16')")
    parser.add_argument("--custom-pipeline", type=str, default=config.DEFAULT_CUSTOM_PIPELINE, help="Custom pipeline class name")
    parser.add_argument("--scheduler", type=str, default=config.DEFAULT_SCHEDULER, help="Scheduler class name")
    parser.add_argument("--lora", type=str, default=config.DEFAULT_LORA, help="Path to LoRA weights to load")
    parser.add_argument("--controlnet", type=str, default=config.DEFAULT_CONTROLNET, help="ControlNet model identifier (HuggingFace ID or local path)")
    parser.add_argument("--steps", type=int, default=config.DEFAULT_STEPS, help="Number of inference steps")
    parser.add_argument("--negative-prompt", type=str, default=config.DEFAULT_NEGATIVE_PROMPT, help="Negative prompt")
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED, help="Random seed for generation")
    parser.add_argument("--warmups", type=int, default=config.DEFAULT_WARMUPS, help="Number of warmup runs before timing")
    parser.add_argument("--batch", type=int, default=config.DEFAULT_BATCH, help="Batch size (number of images per prompt)")
    parser.add_argument("--control-image", type=str, default=config.DEFAULT_CONTROL_IMAGE, help="Path or URL to the control image (for ControlNet)")
    parser.add_argument("--output-image", type=str, default=config.DEFAULT_OUTPUT_IMAGE, help="Path to save the output image (default: auto-generated name)")
    parser.add_argument("--extra-call-kwargs", type=str, default=config.DEFAULT_EXTRA_CALL_KWARGS, help="JSON string of extra kwargs for the pipeline call")
    parser.add_argument("--throughput", action="store_true", help="Run throughput analysis")
    parser.add_argument("--deepcache", action="store_true", help="Enable DeepCache optimization (experimental, primarily for SDXL T2I)")
    parser.add_argument("--cache_interval", type=int, default=config.DEFAULT_CACHE_INTERVAL, help="DeepCache interval")
    parser.add_argument("--cache_layer_id", type=int, default=config.DEFAULT_CACHE_LAYER_ID, help="DeepCache layer ID")
    parser.add_argument("--cache_block_id", type=int, default=config.DEFAULT_CACHE_BLOCK_ID, help="DeepCache block ID")
    parser.add_argument(
        "--compiler",
        type=str,
        default=config.DEFAULT_COMPILER,
        choices=["none", "oneflow", "nexfort", "compile", "compile-max-autotune"],
        help="Compiler backend to use"
    )
    parser.add_argument("--compiler-config", type=str, default=config.DEFAULT_COMPILER_CONFIG, help="JSON string for compiler options (Nexfort)")
    parser.add_argument(
        "--run_multiple_resolutions",
        action="store_true", # Use action='store_true' for boolean flags
        help="Run tests with multiple resolutions after the main run"
    )
    parser.add_argument("--quantize", action="store_true", help="Enable quantization (currently with Nexfort compiler)")
    parser.add_argument("--quantize-config", type=str, default=config.DEFAULT_QUANTIZE_CONFIG, help="JSON string for quantization config")
    parser.add_argument("--quant-submodules-config-path", type=str, default=None, help="Path to quant submodules config JSON (Nexfort)")


def main():
    args = parse_args()

    # Clear cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("Cleared CUDA cache.")

    # Dispatch to the correct pipeline based on the subparser command
    if args.task == 't2i':
        pipelines.text_to_image.run(args)
    elif args.task == 'i2i':
        pipelines.image_to_image.run(args)
    else:
        # This should not happen if subparsers are required, but good practice
        print(f"Error: Unknown task '{args.task}'")
        sys.exit(1)

if __name__ == "__main__":
    main()
