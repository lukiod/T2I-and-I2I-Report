import time
import json
import torch
from PIL import Image, ImageDraw
from diffusers.utils import load_image
import inspect # Added inspect import

# Import shared utilities and config
import utils
import config as base_config # Use alias to avoid name clash

# Import specific pipeline classes
from diffusers import AutoPipelineForText2Image
# Conditional import for deepcache if needed
try:
    from onediffx.deep_cache import StableDiffusionXLPipeline as DeepCacheSDXLPipeline
    deepcache_available = True
except ImportError:
    DeepCacheSDXLPipeline = None # Fallback if deepcache is not installed
    deepcache_available = False

def run(args):
    """Runs the Text-to-Image generation pipeline."""
    print("--- Initializing Text-to-Image Pipeline ---")

    # Determine pipeline class
    if args.deepcache:
        if deepcache_available and args.model and ("xl" in args.model.lower()):
             pipeline_cls = DeepCacheSDXLPipeline
             print("Using DeepCache SDXL Pipeline.")
             # Note: DeepCache might have specific requirements or limitations
        elif not deepcache_available:
             print("Warning: DeepCache requested but 'onediffx.deep_cache' not found. Falling back to standard pipeline.")
             pipeline_cls = AutoPipelineForText2Image
        else:
             print("Warning: DeepCache currently primarily optimized for SDXL models. Using standard pipeline.")
             pipeline_cls = AutoPipelineForText2Image
    else:
        pipeline_cls = AutoPipelineForText2Image
        print(f"Using standard pipeline: {pipeline_cls.__name__}")


    # Load the pipeline using the utility function
    pipe = utils.load_pipe(
        pipeline_cls=pipeline_cls,
        model_name=args.model,
        variant=args.variant,
        custom_pipeline=args.custom_pipeline,
        scheduler=args.scheduler,
        lora=args.lora,
        controlnet=args.controlnet, # Pass controlnet if specified
        dtype=base_config.DEFAULT_DTYPE, # Use dtype from config
        device=base_config.DEFAULT_DEVICE # Use device from config
    )

    # Apply compiler optimizations
    pipe = utils.setup_compiler(
        pipe=pipe,
        compiler_name=args.compiler,
        compiler_config_str=args.compiler_config,
        quantize=args.quantize,
        quantize_config_str=args.quantize_config,
        quant_submodules_config_path=args.quant_submodules_config_path
    )

    # Determine image dimensions
    core_net = utils.get_pipeline_core_net(pipe)
    if core_net:
         # SDXL models often have sample_size * vae_scale_factor = 1024
         # Other models might differ
         default_res = core_net.config.sample_size * pipe.vae_scale_factor
         height = args.height or default_res
         width = args.width or default_res
         print(f"Derived default resolution: {default_res}x{default_res}")
    else:
        # Fallback if core_net couldn't be found
        height = args.height or 1024 # Default fallback resolution
        width = args.width or 1024
        print(f"Warning: Using fallback resolution: {height}x{width}")

    args.height = int(height) # Update args to reflect final dimensions
    args.width = int(width)
    print(f"Using final resolution: {args.height}x{args.width}")

    # --- Prepare Inputs ---
    # ControlNet requires an image input, even for T2I. Create a blank one if needed.
    control_image = None
    if args.controlnet:
        if args.control_image:
            print(f"Loading control image from: {args.control_image}")
            control_image = load_image(args.control_image)
            control_image = control_image.resize((args.width, args.height), Image.LANCZOS)
        else:
            # Create a default blank control image if ControlNet is active but no control image provided
            print("ControlNet is active, but no control image provided. Creating a blank image.")
            control_image = Image.new("RGB", (args.width, args.height), (255, 255, 255)) # White canvas
            # Example: draw a circle
            # draw = ImageDraw.Draw(control_image)
            # draw.ellipse((args.width // 4, args.height // 4, args.width // 4 * 3, args.height // 4 * 3), fill=(128, 128, 128))
            # del draw

    # Kwargs for the pipeline call
    def get_kwarg_inputs():
        kwarg_inputs = dict(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            num_images_per_prompt=args.batch,
            generator=(
                None
                if args.seed is None
                else torch.Generator(device=base_config.DEFAULT_DEVICE).manual_seed(args.seed)
            ),
            **(
                dict()
                if args.extra_call_kwargs is None
                else json.loads(args.extra_call_kwargs)
            ),
        )
        # Add control image if available (for ControlNet)
        # T2I pipelines with ControlNet often expect 'image' or 'control_image'
        if control_image is not None:
             # Check pipeline signature to determine the correct kwarg name
             pipe_signature = inspect.signature(pipe.__call__)
             if "control_image" in pipe_signature.parameters:
                 kwarg_inputs["control_image"] = control_image
             elif "image" in pipe_signature.parameters:
                 print("Using 'image' keyword argument for ControlNet input.")
                 kwarg_inputs["image"] = control_image # Some pipelines use 'image'
             else:
                 print("Warning: Could not determine the correct keyword for control image in pipeline call.")


        # Add deepcache args if enabled
        if args.deepcache:
            kwarg_inputs["cache_interval"] = args.cache_interval
            kwarg_inputs["cache_layer_id"] = args.cache_layer_id
            kwarg_inputs["cache_block_id"] = args.cache_block_id
            # DeepCache might require specific prompt formats or args, check its documentation

        return kwarg_inputs

    # --- Warmup ---
    if args.warmups > 0:
        print(f"\n--- Running {args.warmups} Warmup(s) ---")
        warmup_start_time = time.time()
        for i in range(args.warmups):
            with torch.inference_mode():
                 _ = pipe(**get_kwarg_inputs())
            print(f"Warmup {i+1}/{args.warmups} completed.")
        warmup_end_time = time.time()
        print(f"Warmup finished in {warmup_end_time - warmup_start_time:.3f}s")
        print("------------------------------------")


    # --- Main Inference ---
    print("\n--- Starting Main Inference ---")
    iter_profiler = utils.IterationProfiler()
    kwarg_inputs_final = get_kwarg_inputs()

    # Add callback for step timing if supported by the pipeline
    pipe_signature = inspect.signature(pipe.__call__)
    if "callback_on_step_end" in pipe_signature.parameters:
        kwarg_inputs_final["callback_on_step_end"] = iter_profiler.callback_on_step_end
        print("Iteration profiler attached via callback_on_step_end.")
    elif "callback" in pipe_signature.parameters and "callback_steps" in pipe_signature.parameters:
         kwarg_inputs_final["callback"] = iter_profiler.callback_on_step_end
         kwarg_inputs_final["callback_steps"] = 1 # Call callback every step
         print("Iteration profiler attached via callback.")
    else:
        print("Pipeline does not support step callbacks. Iter/sec measurement will be approximate.")
        iter_profiler.stop() # Disable profiler if callback isn't possible


    inference_start_time = time.time()
    with torch.inference_mode():
         output_images = pipe(**kwarg_inputs_final).images
    inference_end_time = time.time()
    total_inference_time = inference_end_time - inference_start_time
    print("--- Inference Complete ---")

    # --- Reporting ---
    utils.print_summary(args, total_inference_time, iter_profiler, args.compiler)

    # Save output image
    output_filename = utils.prepare_output_filename(args)
    try:
        output_images[0].save(output_filename)
        print(f"Output image saved to: {output_filename}")
    except Exception as e:
        print(f"Error saving output image: {e}")


    # --- Optional: Throughput Analysis ---
    if args.throughput:
         # Prepare kwargs specifically for throughput testing (no callbacks needed)
         throughput_kwargs = get_kwarg_inputs()
         if "callback_on_step_end" in throughput_kwargs: del throughput_kwargs["callback_on_step_end"]
         if "callback" in throughput_kwargs: del throughput_kwargs["callback"]
         if "callback_steps" in throughput_kwargs: del throughput_kwargs["callback_steps"]

         steps_range = range(10, 61, 10) # Example range: 10, 20, 30, 40, 50, 60 steps
         data, coefficients = utils.generate_data_and_fit_model(
             pipe=pipe,
             prompt=args.prompt, # Use the main prompt
             height=args.height,
             width=args.width,
             steps_range=steps_range,
             extra_pipe_kwargs=throughput_kwargs # Pass other relevant args
         )
         utils.plot_data_and_model(data, coefficients, title="T2I Inference Time vs. Steps")

    # --- Optional: Multiple Resolutions Test ---
    if args.run_multiple_resolutions:
        print("\n--- Testing Multiple Resolutions ---")
        # Define resolutions to test, ensure they are valid for the model
        sizes = [512, 768, 1024] # Example sizes, adjust as needed
        res_kwargs = get_kwarg_inputs() # Get base kwargs
        if "callback_on_step_end" in res_kwargs: del res_kwargs["callback_on_step_end"]
        if "callback" in res_kwargs: del res_kwargs["callback"]
        if "callback_steps" in res_kwargs: del res_kwargs["callback_steps"]


        for h in sizes:
            for w in sizes:
                 # Check if resolution is compatible (e.g., multiple of 8 or 64)
                 if h % 8 != 0 or w % 8 != 0:
                     print(f"Skipping resolution {h}x{w} (not divisible by 8)")
                     continue

                 res_kwargs["height"] = h
                 res_kwargs["width"] = w
                 print(f"Running at resolution: {h}x{w}...")
                 start_time = time.time()
                 try:
                    with torch.inference_mode():
                         _ = pipe(**res_kwargs).images
                    end_time = time.time()
                    print(f"  Inference time: {end_time - start_time:.2f} seconds")
                 except Exception as e:
                     print(f"  Error running at {h}x{w}: {e}")
        print("------------------------------------")
