import time
import json
import torch
from PIL import Image, ImageDraw
from diffusers.utils import load_image
import inspect

# Import shared utilities and config
import utils
import config as base_config # Use alias

# Import specific pipeline class for InstructPix2Pix
from diffusers import StableDiffusionInstructPix2PixPipeline

def run(args):
    """Runs the Image-to-Image (InstructPix2Pix) generation pipeline."""
    print("--- Initializing Image-to-Image (InstructPix2Pix) Pipeline ---")

    # Specific pipeline class for this task
    pipeline_cls = StableDiffusionInstructPix2PixPipeline
    print(f"Using pipeline: {pipeline_cls.__name__}")

    # Load the pipeline using the utility function
    pipe = utils.load_pipe(
        pipeline_cls=pipeline_cls,
        model_name=args.model,
        variant=args.variant,
        custom_pipeline=args.custom_pipeline,
        scheduler=args.scheduler,
        lora=args.lora,
        # Note: InstructPix2Pix doesn't typically use ControlNet in the same way
        # If you need ControlNet with I2I, you might need a different pipeline (e.g., StableDiffusionControlNetImg2ImgPipeline)
        # For now, assuming ControlNet is not used with InstructPix2Pix here.
        controlnet=None, # Explicitly None for InstructPix2Pix
        dtype=base_config.DEFAULT_DTYPE,
        device=base_config.DEFAULT_DEVICE
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

    # Determine image dimensions (I2I often uses fixed input size)
    # Use args directly as they are required/have defaults for I2I
    height = args.height
    width = args.width
    args.height = int(height) # Ensure they are integers
    args.width = int(width)
    print(f"Using resolution: {args.height}x{args.width}")

    # --- Prepare Inputs ---
    # Input image is REQUIRED for I2I
    if not args.input_image:
        raise ValueError("--input-image is required for the image-to-image task.")

    print(f"Loading input image from: {args.input_image}")
    try:
        input_image = load_image(args.input_image)
        # Resize input image to the target dimensions
        input_image = input_image.resize((args.width, args.height), Image.LANCZOS)
        print(f"Input image resized to {args.width}x{args.height}")
    except Exception as e:
        raise RuntimeError(f"Failed to load or process input image: {e}")


    # Kwargs for the pipeline call
    def get_kwarg_inputs():
        kwarg_inputs = dict(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            image=input_image, # Pass the loaded and resized input image
            # InstructPix2Pix might not use height/width args directly if image is provided
            # Check the pipeline's signature if needed. Typically they infer from image.
            # height=args.height, # Often not needed if image is passed
            # width=args.width,   # Often not needed if image is passed
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
            # Add specific InstructPix2Pix parameters if needed (e.g., image_guidance_scale)
            # Check inspect.signature(pipe.__call__) or documentation
        )
        # Example: Add common I2I parameters if they exist in args
        if hasattr(args, 'guidance_scale'):
             kwarg_inputs['guidance_scale'] = args.guidance_scale
        if hasattr(args, 'image_guidance_scale'): # Specific to InstructPix2Pix
            kwarg_inputs['image_guidance_scale'] = args.image_guidance_scale


        # Add deepcache args if enabled (check if compatible with InstructPix2Pix)
        # Compatibility not guaranteed, may need specific DeepCache version or pipeline
        if args.deepcache:
             print("Warning: DeepCache compatibility with InstructPix2Pix is experimental.")
             kwarg_inputs["cache_interval"] = args.cache_interval
             kwarg_inputs["cache_layer_id"] = args.cache_layer_id
             kwarg_inputs["cache_block_id"] = args.cache_block_id

        # Filter out None values just in case
        kwarg_inputs = {k: v for k, v in kwarg_inputs.items() if v is not None}
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

    # Add callback for step timing
    pipe_signature = inspect.signature(pipe.__call__)
    if "callback_on_step_end" in pipe_signature.parameters:
        kwarg_inputs_final["callback_on_step_end"] = iter_profiler.callback_on_step_end
        print("Iteration profiler attached via callback_on_step_end.")
    elif "callback" in pipe_signature.parameters and "callback_steps" in pipe_signature.parameters:
         kwarg_inputs_final["callback"] = iter_profiler.callback_on_step_end
         kwarg_inputs_final["callback_steps"] = 1
         print("Iteration profiler attached via callback.")
    else:
        print("Pipeline does not support step callbacks. Iter/sec measurement will be approximate.")
        iter_profiler.stop()


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
         # Prepare kwargs specifically for throughput testing
         throughput_kwargs = get_kwarg_inputs()
         if "callback_on_step_end" in throughput_kwargs: del throughput_kwargs["callback_on_step_end"]
         if "callback" in throughput_kwargs: del throughput_kwargs["callback"]
         if "callback_steps" in throughput_kwargs: del throughput_kwargs["callback_steps"]

         steps_range = range(10, 61, 10) # Example range
         data, coefficients = utils.generate_data_and_fit_model(
             pipe=pipe,
             prompt=args.prompt, # Use the main prompt
             height=args.height, # Pass height/width even if unused by pipe directly, needed for reporting/consistency
             width=args.width,
             steps_range=steps_range,
             extra_pipe_kwargs=throughput_kwargs # Pass other relevant args like 'image'
         )
         utils.plot_data_and_model(data, coefficients, title="I2I (InstructPix2Pix) Inference Time vs. Steps")

    # --- Optional: Multiple Resolutions Test ---
    # Note: I2I models often expect a specific input size. Testing multiple resolutions might require
    # resizing the *input* image each time and may not be standard practice unless the model supports it well.
    # If you want to test this, uncomment and adapt.
    # if args.run_multiple_resolutions:
    #     print("\n--- Testing Multiple Resolutions (I2I) ---")
    #     # Define resolutions to test
    #     sizes = [256, 512, 768] # Example sizes, ensure compatibility
    #     base_input_image = load_image(args.input_image) # Load original once
    #     res_kwargs = get_kwarg_inputs() # Get base kwargs
    #     if "callback_on_step_end" in res_kwargs: del res_kwargs["callback_on_step_end"]
    #     if "callback" in res_kwargs: del res_kwargs["callback"]
    #     if "callback_steps" in res_kwargs: del res_kwargs["callback_steps"]
    #
    #     for size in sizes:
    #         h, w = size, size # Assuming square images for simplicity
    #         if h % 8 != 0 or w % 8 != 0:
    #              print(f"Skipping resolution {h}x{w} (not divisible by 8)")
    #              continue
    #
    #         print(f"Running at resolution: {h}x{w}...")
    #         # Resize the input image for each test
    #         resized_input = base_input_image.resize((w, h), Image.LANCZOS)
    #         res_kwargs["image"] = resized_input
    #         # Update height/width if needed by the specific pipeline call signature
    #         # res_kwargs["height"] = h
    #         # res_kwargs["width"] = w
    #         start_time = time.time()
    #         try:
    #             with torch.inference_mode():
    #                 _ = pipe(**res_kwargs).images
    #             end_time = time.time()
    #             print(f"  Inference time: {end_time - start_time:.2f} seconds")
    #         except Exception as e:
    #             print(f"  Error running at {h}x{w}: {e}")
    #     print("------------------------------------")
