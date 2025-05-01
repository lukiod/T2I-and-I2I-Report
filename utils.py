import os
import importlib
import inspect
import time
import json
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from diffusers.utils import load_image

from onediffx import compile_pipe, quantize_pipe # quantize_pipe currently only supports the nexfort backend.
import config # Import defaults from config.py

def load_pipe(
    pipeline_cls,
    model_name,
    variant=None,
    dtype=config.DEFAULT_DTYPE,
    device=config.DEFAULT_DEVICE,
    custom_pipeline=None,
    scheduler=None,
    lora=None,
    controlnet=None,
):
    """Loads the diffusion pipeline."""
    extra_kwargs = {}
    if custom_pipeline is not None:
        extra_kwargs["custom_pipeline"] = custom_pipeline
    if variant is not None:
        extra_kwargs["variant"] = variant
    if dtype is not None:
        extra_kwargs["torch_dtype"] = dtype
    if controlnet is not None:
        from diffusers import ControlNetModel
        controlnet = ControlNetModel.from_pretrained(
            controlnet,
            torch_dtype=dtype,
        )
        extra_kwargs["controlnet"] = controlnet

    model_path = model_name # Assume model_name is a path or HF identifier
    # Check if it's a local path and quantization info exists
    if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "calibrate_info.txt")):
        print(f"Detected quantization info in {model_path}. Loading quantized pipeline.")
        from onediff.quantization import QuantPipeline
        pipe = QuantPipeline.from_quantized(pipeline_cls, model_path, **extra_kwargs)
    else:
        print(f"Loading standard pipeline from {model_path}.")
        pipe = pipeline_cls.from_pretrained(model_name, **extra_kwargs)

    if scheduler is not None and scheduler != "none":
        try:
            scheduler_cls = getattr(importlib.import_module("diffusers"), scheduler)
            pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)
            print(f"Applied scheduler: {scheduler}")
        except AttributeError:
            print(f"Warning: Could not find scheduler '{scheduler}' in diffusers. Using default.")
    if lora is not None:
        print(f"Loading LoRA weights from {lora}")
        pipe.load_lora_weights(lora)
        print("Fusing LoRA weights.")
        pipe.fuse_lora()
    pipe.safety_checker = None
    if device is not None:
        print(f"Moving pipeline to device: {device}")
        pipe.to(torch.device(device))
    return pipe


def setup_compiler(pipe, compiler_name, compiler_config_str, quantize, quantize_config_str, quant_submodules_config_path):
    """Applies compiler and quantization options to the pipeline."""
    if compiler_name == "none":
        print("No compiler selected.")
        pass # No compilation
    elif compiler_name == "oneflow":
        print("Applying OneFlow compiler...")
        pipe = compile_pipe(pipe)
    elif compiler_name == "nexfort":
        print("Applying Nexfort compiler...")
        if quantize:
            print("Quantization enabled for Nexfort.")
            if quantize_config_str:
                try:
                    quantize_config = json.loads(quantize_config_str)
                    print(f"Using custom quantization config: {quantize_config}")
                except json.JSONDecodeError:
                    print(f"Error parsing quantize-config JSON. Using default.")
                    quantize_config = {"quant_type": "fp8_e4m3_e4m3_dynamic"} # Default fallback
            else:
                quantize_config = {"quant_type": "fp8_e4m3_e4m3_dynamic"} # Default
                print(f"Using default quantization config: {quantize_config}")

            if quant_submodules_config_path:
                 print(f"Using quant submodules config path: {quant_submodules_config_path}")
                 # Ensure 'ignores' key exists if needed by quantize_pipe
                 if 'ignores' not in quantize_config:
                     quantize_config['ignores'] = []
                 pipe = quantize_pipe(
                    pipe,
                    quant_submodules_config_path=quant_submodules_config_path,
                    **quantize_config,
                )
            else:
                 # Ensure 'ignores' key exists if needed by quantize_pipe
                 if 'ignores' not in quantize_config:
                     quantize_config['ignores'] = []
                 pipe = quantize_pipe(pipe, **quantize_config)

        if compiler_config_str:
             try:
                 options = json.loads(compiler_config_str)
                 print(f"Using custom compiler options: {options}")
             except json.JSONDecodeError:
                 print(f"Error parsing compiler-config JSON. Using default.")
                 options = {"mode": "max-optimize:max-autotune:freezing", "memory_format": "channels_last"} # Default fallback
        else:
             options = {"mode": "max-optimize:max-autotune:freezing", "memory_format": "channels_last"} # Default
             print(f"Using default compiler options: {options}")

        pipe = compile_pipe(
            pipe, backend="nexfort", options=options, fuse_qkv_projections=True
        )
    elif compiler_name in ("compile", "compile-max-autotune"):
        mode = "max-autotune" if compiler_name == "compile-max-autotune" else None
        print(f"Applying torch.compile (mode: {mode})...")
        # Compile components that are commonly present
        compiled_components = []
        if hasattr(pipe, "unet") and pipe.unet is not None:
            print("Compiling unet...")
            pipe.unet = torch.compile(pipe.unet, mode=mode)
            compiled_components.append("unet")
        if hasattr(pipe, "transformer") and pipe.transformer is not None:
             print("Compiling transformer...")
             pipe.transformer = torch.compile(pipe.transformer, mode=mode)
             compiled_components.append("transformer")
        if hasattr(pipe, "controlnet") and pipe.controlnet is not None:
            print("Compiling controlnet...")
            pipe.controlnet = torch.compile(pipe.controlnet, mode=mode)
            compiled_components.append("controlnet")
        if hasattr(pipe, "vae") and pipe.vae is not None:
             print("Compiling vae...")
             pipe.vae = torch.compile(pipe.vae, mode=mode)
             compiled_components.append("vae")
        if not compiled_components:
            print("Warning: No standard components (unet, transformer, controlnet, vae) found to compile.")
        else:
            print(f"Compiled components: {', '.join(compiled_components)}")

    else:
        raise ValueError(f"Unknown compiler: {compiler_name}")
    return pipe


class IterationProfiler:
    """Profiles iterations per second during pipeline execution."""
    def __init__(self):
        self.begin = None
        self.end = None
        self.num_iterations = 0
        self.enabled = True # Control whether profiling is active

    def start(self):
        """Reset and enable profiling."""
        self.begin = None
        self.end = None
        self.num_iterations = 0
        self.enabled = True

    def stop(self):
        """Disable profiling."""
        self.enabled = False

    def get_iter_per_sec(self):
        if not self.enabled or self.begin is None or self.end is None or self.num_iterations == 0:
            return None
        # Ensure CUDA synchronization to get accurate timing
        if isinstance(self.begin, torch.cuda.Event) and isinstance(self.end, torch.cuda.Event):
            self.end.synchronize()
            dur = self.begin.elapsed_time(self.end) # Time in milliseconds
            return self.num_iterations / dur * 1000.0
        else:
            # Fallback for non-CUDA or initial state - less precise
            return self.num_iterations / (self.end - self.begin) if (self.end - self.begin) > 0 else 0

    def callback_on_step_end(self, pipe, i, t, callback_kwargs={}):
        """Callback function for diffusion pipeline steps."""
        if not self.enabled:
            return callback_kwargs

        if torch.cuda.is_available():
            # Use CUDA events for accurate timing on GPU
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            if self.begin is None:
                self.begin = event # Record start event
            else:
                self.end = event # Record end event for the iteration
                self.num_iterations += 1
        else:
            # Use time.time() as a fallback for CPU or if CUDA events fail
            current_time = time.time()
            if self.begin is None:
                self.begin = current_time
            else:
                self.end = current_time
                self.num_iterations += 1
        return callback_kwargs


def calculate_inference_time_and_throughput(pipe, prompt, height, width, n_steps, extra_pipe_kwargs={}):
    """Measures inference time and throughput for a single run."""
    start_time = time.time()
    # Create a minimal set of required args, merge with extra ones
    call_args = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "num_inference_steps": n_steps,
        **extra_pipe_kwargs # Add other necessary args like 'image' if needed
    }
    # Filter out None values as pipe might not accept them
    call_args = {k: v for k, v in call_args.items() if v is not None}
    _ = pipe(**call_args).images
    end_time = time.time()
    inference_time = end_time - start_time
    throughput = n_steps / inference_time if inference_time > 0 else 0
    return inference_time, throughput

def generate_data_and_fit_model(pipe, prompt, height, width, steps_range, extra_pipe_kwargs={}):
    """Generates throughput data across a range of steps and fits a linear model."""
    data = {"steps": [], "inference_time": [], "throughput": []}

    print(f"\n--- Throughput Analysis ({height}x{width}) ---")
    for n_steps in steps_range:
        try:
            inference_time, throughput = calculate_inference_time_and_throughput(
                pipe, prompt, height, width, n_steps, extra_pipe_kwargs
            )
            data["steps"].append(n_steps)
            data["inference_time"].append(inference_time)
            data["throughput"].append(throughput)
            print(
                f"Steps: {n_steps:3d}, Inference Time: {inference_time:6.2f}s, Throughput: {throughput:6.2f} steps/s"
            )
        except Exception as e:
            print(f"Error during throughput test at {n_steps} steps: {e}")
            # Optionally break or continue
            break

    if not data["steps"]:
        print("No data collected for throughput analysis.")
        return data, None

    average_throughput = np.mean(data["throughput"])
    print(f"\nAverage Throughput: {average_throughput:.2f} steps/s")

    # Fit a linear model: time = slope * steps + intercept
    try:
        coefficients = np.polyfit(data["steps"], data["inference_time"], 1)
        slope = coefficients[0]
        intercept = coefficients[1]
        if slope > 0:
             throughput_no_base = 1 / slope
             print(f"Base time (intercept): {intercept:.2f}s")
             print(f"Throughput ignoring base cost (1/slope): {throughput_no_base:.2f} steps/s")
        else:
             print("Warning: Non-positive slope detected in time vs steps. Cannot calculate base cost adjusted throughput.")
             coefficients = None # Indicate model fitting failed meaningfully
    except np.linalg.LinAlgError:
        print("Linear regression failed. Cannot calculate base cost adjusted throughput.")
        coefficients = None


    print("-------------------------------------\n")
    return data, coefficients

def plot_data_and_model(data, coefficients, title="Inference Time vs. Steps"):
    """Plots the collected throughput data and the fitted linear model."""
    if not data["steps"]:
        print("No data to plot.")
        return

    plt.figure(figsize=(10, 5))
    plt.scatter(data["steps"], data["inference_time"], color="blue", label="Measured Data")
    if coefficients is not None:
        plt.plot(data["steps"], np.polyval(coefficients, data["steps"]), color="red", label=f"Linear Fit: Time = {coefficients[0]:.3f}*Steps + {coefficients[1]:.3f}")
        plt.legend()

    plt.title(title)
    plt.xlabel("Number of Inference Steps")
    plt.ylabel("Inference Time (seconds)")
    plt.grid(True)
    # Consider saving instead of showing:
    # plt.savefig("throughput_analysis.png")
    # print("Throughput plot saved as throughput_analysis.png")
    plt.show()


def get_pipeline_core_net(pipe):
    """Attempts to find the main neural network (unet or transformer) in a pipeline."""
    if hasattr(pipe, "unet") and pipe.unet is not None:
        return pipe.unet
    elif hasattr(pipe, "transformer") and pipe.transformer is not None:
         return pipe.transformer
    else:
        print("Warning: Could not automatically determine the core network (unet/transformer). Default resolution might be incorrect.")
        return None

def prepare_output_filename(args):
    """Creates a default output filename if none is provided."""
    if args.output_image:
        return args.output_image

    # Create a filename based on task and key parameters
    task_name = args.task if hasattr(args, 'task') else "unknown_task"
    model_part = os.path.basename(args.model.rstrip('/')) if args.model else "default_model"
    compiler_part = args.compiler
    ts = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{task_name}_{model_part}_{compiler_part}_{ts}.png"
    print(f"Output image filename not specified. Using default: {filename}")
    return filename

def print_summary(args, inference_time, iter_profiler, compiler_name):
    """Prints a summary of the run including performance metrics."""
    print("\n================ Run Summary ================")
    print(f"Task:               {args.task.upper()}")
    print(f"Model:              {args.model}")
    if args.variant:
        print(f"Variant:            {args.variant}")
    print(f"Steps:              {args.steps}")
    print(f"Resolution:         {args.height}x{args.width}")
    print(f"Batch Size:         {args.batch}")
    print(f"Seed:               {args.seed}")
    print(f"Compiler:           {compiler_name}")
    if args.quantize:
        print(f"Quantization:       Enabled (Config: {args.quantize_config or 'Default'})")
    print("-------------------------------------------")
    print(f"Total Inference time: {inference_time:.3f}s")
    iter_per_sec = iter_profiler.get_iter_per_sec()
    if iter_per_sec is not None:
        print(f"Avg. Iter/sec:      {iter_per_sec:.3f}")

    # Memory reporting
    if torch.cuda.is_available():
        if compiler_name == "oneflow":
            try:
                import oneflow as flow # Keep import local if only used here
                # Note: GetCUDAMemoryUsed might report differently than torch
                cuda_mem_used = flow._oneflow_internal.GetCUDAMemoryUsed() / 1024 # KiB
                # OneFlow doesn't have a direct equivalent to max_memory_allocated easily accessible
                # Report current used memory as an indicator
                print(f"Current CUDA Mem Used (OneFlow): {cuda_mem_used / (1024**2):.3f} GiB")
            except ImportError:
                 print("OneFlow not found, cannot report OneFlow memory usage.")
            except Exception as e:
                 print(f"Could not get OneFlow memory usage: {e}")

        else: # PyTorch or other backends
             # max_memory_allocated reports the peak usage since start or last reset_peak_stats
             max_mem_gib = torch.cuda.max_memory_allocated() / (1024**3)
             print(f"Peak CUDA Memory Used: {max_mem_gib:.3f} GiB")
             # Reset stats for potential subsequent runs if needed
             # torch.cuda.reset_peak_memory_stats()
    else:
        print("CUDA not available, memory usage not reported.")
    print("===========================================\n")
