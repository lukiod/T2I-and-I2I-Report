# T2I-and-I2I-Report
# What is OneDiff
OneDiff is an acceleration library for diffusion models, allowing for faster inference with minimal code changes. The name stands 	for "one line of code to accelerate diffusion models". It achieves this through features like PyTorch Module compilation and optimised GPU Kernels. 


## The Research Question

Does OneDiff improve the inference speed of our T2I and I2I pipelines while still providing high quality.

## Method
To evaluate the performance impact of OneDiff optimization, I conducted a series of tests using the provided benchmarking script. The methodology involved running the script multiple times for each model, with the number of inference steps consistently set to 10 across all runs. I specifically utilized the Nexfort compiler for all operations, as specified in the script. The testing process involved comparing the performance of runs with and without OneDiff optimization. For each run, I measured and recorded the inference time. To ensure fair comparisons, I maintained consistent parameters across all runs, including the number of inference steps and other relevant settings. This approach allowed for a structured assessment of the performance differences between standard model execution and the OneDiff-optimized version, all while leveraging the Nexfort compiler.
## Make the codes work
### Dependencies
`python3 -m  pip install -U torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 torchao==0.1`

`python3 -m  pip install -U nexfort`

`pip install onediffx`

### How to run the file independently
If you want to run with onediff use compiler nexfort otherwise none
#### Text 2 Image Conversion
`python3 ./text_to_image.py  --scheduler none --steps 10 --height 1024 --width 1024 --compiler none --compiler-config '{"mode": "max-optimize:max-autotune:low-precision", "memory_format": "channels_last", "dynamic": true}' --output-image ./test.png`
<details>
	<summary>Code Details</summary>

### Script used for model usage using nexfort
```
 print("Nexfort backend is now active...")
        if args.quantize:
            if args.quantize_config is not None:
                quantize_config = json.loads(args.quantize_config)
            else:
                quantize_config = '{"quant_type": "fp8_e4m3_e4m3_dynamic"}'
            if args.quant_submodules_config_path:
                # download: https://huggingface.co/siliconflow/PixArt-alpha-onediff-nexfort-fp8/blob/main/fp8_e4m3.json
                pipe = quantize_pipe(
                    pipe,
                    quant_submodules_config_path=args.quant_submodules_config_path,
                    ignores=[],
                    **quantize_config,
                )
            else:
                pipe = quantize_pipe(pipe, ignores=[], **quantize_config)
        if args.compiler_config is not None:
            # config with dict
            options = json.loads(args.compiler_config)
        else:
            # config with string
            options = '{"mode": "max-optimize:max-autotune:freezing", "memory_format": "channels_last"}'
        pipe = compile_pipe(
            pipe, backend="nexfort", options=options, fuse_qkv_projections=True
        ) print("Nexfort backend is now active...")
        if args.quantize:
            if args.quantize_config is not None:
                quantize_config = json.loads(args.quantize_config)
            else:
                quantize_config = '{"quant_type": "fp8_e4m3_e4m3_dynamic"}'
            if args.quant_submodules_config_path:
                # download: https://huggingface.co/siliconflow/PixArt-alpha-onediff-nexfort-fp8/blob/main/fp8_e4m3.json
                pipe = quantize_pipe(
                    pipe,
                    quant_submodules_config_path=args.quant_submodules_config_path,
                    ignores=[],
                    **quantize_config,
                )
            else:
                pipe = quantize_pipe(pipe, ignores=[], **quantize_config)
        if args.compiler_config is not None:
            # config with dict
            options = json.loads(args.compiler_config)
        else:
            # config with string
            options = '{"mode": "max-optimize:max-autotune:freezing", "memory_format": "channels_last"}'
        pipe = compile_pipe(
            pipe, backend="nexfort", options=options, fuse_qkv_projections=True
        )
```
	
### Default Attributes:

```
MODEL = "SG161222/RealVisXL_V4.0"
VARIANT = None
CUSTOM_PIPELINE = None
SCHEDULER = "EulerAncestralDiscreteScheduler"
LORA = None
CONTROLNET = None
STEPS = 30
PROMPT = "best quality, realistic, unreal engine, 4K,a cat sitting on human lap"
NEGATIVE_PROMPT = ""
SEED = 333
WARMUPS = 1
BATCH = 1
HEIGHT = None
WIDTH = None
INPUT_IMAGE = None
CONTROL_IMAGE = None
OUTPUT_IMAGE = None
EXTRA_CALL_KWARGS = None
CACHE_INTERVAL = 3
CACHE_LAYER_ID = 0
CACHE_BLOCK_ID = 0
COMPILER = "nexfort"
COMPILER_CONFIG = None
QUANTIZE_CONFIG = None
```
● This code block defines a function parse_args to handle command-line arguments without using any attributes.


### Argument Parsing and Configuration:
	
```
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--variant", type=str, default=VARIANT)
    parser.add_argument("--custom-pipeline", type=str, default=CUSTOM_PIPELINE)
    parser.add_argument("--scheduler", type=str, default=SCHEDULER)
    # ... other argument definitions
    return parser.parse_args()
```

`args = parse_args()`
● This code block defines a function parse_args to handle command-line arguments using argparse.It defines various arguments such as model, variant, custom-pipeline, scheduler, etc., each with a default value from globally defined variables.This allows users to customize the text-to-image generation process from the command line.The line `args = parse_args()` calls the function and stores the parsed arguments in the args variable for later use.

###  Pipeline Loading and Configuration:
```
def load_pipe(
    pipeline_cls,
    model_name,
    variant=None,
    dtype=torch.float16,
    device="cuda",
    custom_pipeline=None,
    scheduler=None,
    lora=None,
    controlnet=None,
):
    # ... function implementation ...
```
● This code defines a function load_pipe that is responsible for loading and configuring the text-to-image generation pipeline.●
It takes several arguments including the pipeline class (pipeline_cls), model name (model_name), variant, data type (dtype), device, and optional components like a custom pipeline, scheduler, LoRA (Low-Rank Adaptation), and ControlNet.●
The function handles loading the pre-trained model, potentially applying quantization, setting up the scheduler, loading LoRA weights, and moving the pipeline to the specified device.
### Inference Time and Throughput Calculation:
```
def calculate_inference_time_and_throughput(height, width, n_steps, model):
    start_time = time.time()
    model(prompt=args.prompt, height=height, width=width, num_inference_steps=n_steps)
    end_time = time.time()
    inference_time = end_time - start_time
    # pixels_processed = height * width * n_steps
    # throughput = pixels_processed / inference_time
    throughput = n_steps / inference_time
    return inference_time, throughput
```
● This code defines a function calculate_inference_time_and_throughput to measure the performance of the text-to-image generation process.It takes the image height, width, number of inference steps, and the model as input.The function records the start and end time of the generation process to calculate the inference time.Throughput is then calculated as the number of steps per second.

### Keyword Argument Handling:
```
def get_kwarg_inputs():
    kwarg_inputs = dict(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=height,
        width=width,
        # ... other keyword arguments ...
    )
    # ... additional argument handling ...
    return kwarg_inputs
```
●This code defines a function get_kwarg_inputs to collect and organize keyword arguments that will be passed to the text-to-image generation pipeline.It gathers arguments such as prompt, negative_prompt, height, width, and others, which control the generation process.The function handles optional arguments like the input image, control image, deep caching options, and additional arguments from the extra_call_kwargs variable.

### Performance Profiling
The IterationProfiler class and related functions measure performance:
```
class IterationProfiler:
    def __init__(self):
        self.begin = None
        self.end = None
        self.num_iterations = 0

    # ... (methods for profiling)
```


### Main Method details
#### Pipeline Loading:
```
pipe = load_pipe(
    pipeline_cls,
    args.model,
    variant=args.variant,
    custom_pipeline=args.custom_pipeline,
    scheduler=args.scheduler,
    lora=args.lora,
    controlnet=args.controlnet,
)
```
This loads the specified diffusion model pipeline with various customization options like variant, custom pipeline, scheduler, LoRA, and ControlNet.

#### Image Size Determination:

```
height = args.height or core_net.config.sample_size * pipe.vae_scale_factor
width = args.width or core_net.config.sample_size * pipe.vae_scale_factor
```
Sets the output image dimensions, either from user arguments or based on the model's default configuration.

#### Compiler Optimization:

```
if args.compiler == "none":
    pass
elif args.compiler == "oneflow":
    pipe = compile_pipe(pipe)
elif args.compiler == "nexfort":
    # ... (nexfort compilation logic)
elif args.compiler in ("compile", "compile-max-autotune"):
    # ... (torch.compile logic)
```
Applies various compiler optimizations to the pipeline based on the specified compiler option.

#### Input Image Handling:

```
if args.input_image is None:
    input_image = None
else:
    input_image = load_image(args.input_image)
    input_image = input_image.resize((width, height), Image.LANCZOS)
```
Loads and resizes an input image if specified (for image-to-image tasks).

 #### Control Image Handling:

```
if args.control_image is None:
    if args.controlnet is None:
        control_image = None
    else:
        # ... (create a default control image)
else:
    control_image = load_image(args.control_image)
    control_image = control_image.resize((width, height), Image.LANCZOS)
```
Prepares a control image for ControlNet, either loading a specified image or creating a default one.

#### Warm-up Runs:

```
if args.warmups > 0:
    # ... (perform warm-up runs)
```
Executes warm-up runs to trigger compilation and initial optimizations.As for Warmup is basically the time taken to bring the model to its full capable loading state

#### Main Inference:

```
kwarg_inputs = get_kwarg_inputs()
iter_profiler = IterationProfiler()
# ... (set up profiling callback)
begin = time.time()
output_images = pipe(**kwarg_inputs).images
end = time.time()
```
Performs the main image generation inference, with profiling.

#### Performance Reporting:

```
print(f"Inference time: {end - begin:.3f}s")
iter_per_sec = iter_profiler.get_iter_per_sec()
if iter_per_sec is not None:
    print(f"Iterations per second: {iter_per_sec:.3f}")
# ... (memory usage reporting)
```
Reports various performance metrics like inference time, iterations per second, and memory usage.

#### Output Image Saving:

```
if args.output_image is not None:
    output_images[0].save(args.output_image)
Saves the generated image if an output path is specified.
```

#### Multi-resolution Testing:

```
if args.run_multiple_resolutions:
    # ... (run inference at multiple resolutions)
```
Tests the model's performance across various image resolutions. As for i wont recommend running this as for i made various testing changes and now code is barely linked as to work properly this might be broken.

#### Throughput Analysis:

```
if args.throughput:
    steps_range = range(1, 100, 1)
    data, coefficients = generate_data_and_fit_model(pipe, steps_range)
    plot_data_and_model(data, coefficients)
```
If requested, performs a detailed throughput analysis across different numbers of inference steps and plots the results.
This is the main blocks where all the things are handled and you will be able to understand most of the features used in the code using this block.

### Main Execution Block:
```
if __name__ == "__main__":
    main()
```
●This is a common Python idiom. It ensures that the main() function is called only when the script is run directly, not when it's imported as a module.
This documentation provides a breakdown of the code snippets, explaining their purpose and how they fit into the larger text-to-image generation process.

</details>

#### Image 2 Image Conversion
`python3 testi2i.py --input-image ./RealVisXL_withoutonediff_1024.png --height 1024 --width 1024 --compiler none --output-image ./i2i_1024__timebrooks_withoutonediff.png --prompt "turn it into a painting painted by paintbrush"`
<details>
	<summary>Code Details</summary>

 ### Script used for model usage using nexfort
```
 print("Nexfort backend is now active...")
        if args.quantize:
            if args.quantize_config is not None:
                quantize_config = json.loads(args.quantize_config)
            else:
                quantize_config = '{"quant_type": "fp8_e4m3_e4m3_dynamic"}'
            if args.quant_submodules_config_path:
                # download: https://huggingface.co/siliconflow/PixArt-alpha-onediff-nexfort-fp8/blob/main/fp8_e4m3.json
                pipe = quantize_pipe(
                    pipe,
                    quant_submodules_config_path=args.quant_submodules_config_path,
                    ignores=[],
                    **quantize_config,
                )
            else:
                pipe = quantize_pipe(pipe, ignores=[], **quantize_config)
        if args.compiler_config is not None:
            # config with dict
            options = json.loads(args.compiler_config)
        else:
            # config with string
            options = '{"mode": "max-optimize:max-autotune:freezing", "memory_format": "channels_last"}'
        pipe = compile_pipe(
            pipe, backend="nexfort", options=options, fuse_qkv_projections=True
        ) print("Nexfort backend is now active...")
        if args.quantize:
            if args.quantize_config is not None:
                quantize_config = json.loads(args.quantize_config)
            else:
                quantize_config = '{"quant_type": "fp8_e4m3_e4m3_dynamic"}'
            if args.quant_submodules_config_path:
                # download: https://huggingface.co/siliconflow/PixArt-alpha-onediff-nexfort-fp8/blob/main/fp8_e4m3.json
                pipe = quantize_pipe(
                    pipe,
                    quant_submodules_config_path=args.quant_submodules_config_path,
                    ignores=[],
                    **quantize_config,
                )
            else:
                pipe = quantize_pipe(pipe, ignores=[], **quantize_config)
        if args.compiler_config is not None:
            # config with dict
            options = json.loads(args.compiler_config)
        else:
            # config with string
            options = '{"mode": "max-optimize:max-autotune:freezing", "memory_format": "channels_last"}'
        pipe = compile_pipe(
            pipe, backend="nexfort", options=options, fuse_qkv_projections=True
        )
```
	
### Default Attributes:
```
MODEL = "timbrooks/instruct-pix2pix"
VARIANT = None
CUSTOM_PIPELINE = None
SCHEDULER = "EulerAncestralDiscreteScheduler"
LORA = None
CONTROLNET = None
STEPS = 30
PROMPT = "make "
NEGATIVE_PROMPT = ""
SEED = 333
WARMUPS = 1
BATCH = 1
HEIGHT = 512
WIDTH = 512
INPUT_IMAGE = "https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg"  # Set a default input image path
CONTROL_IMAGE = None
OUTPUT_IMAGE = None
EXTRA_CALL_KWARGS = None
CACHE_INTERVAL = 3
CACHE_LAYER_ID = 0
CACHE_BLOCK_ID = 0
COMPILER = "nexfort"
COMPILER_CONFIG = None
QUANTIZE_CONFIG = None
```
● This code block defines default values for various parameters used in the text-to-image generation process. These values serve as fallbacks if not specified by the user.
### Argument Parsing and Configuration:
```
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--variant", type=str, default=VARIANT)
    parser.add_argument("--custom-pipeline", type=str, default=CUSTOM_PIPELINE)
    parser.add_argument("--scheduler", type=str, default=SCHEDULER)
    # ... other argument definitions
    return parser.parse_args()
```
`args = parse_args()`
● This code defines the parse_args function to handle command-line arguments using argparse. It allows users to customize various aspects of the text-to-image generation process. The parsed arguments are stored in the args variable for later use throughout the script.
### Pipeline Loading and Configuration:
```
def load_pipe(
    pipeline_cls,
    model_name,
    variant=None,
    dtype=torch.float16,
    device="cuda",
    custom_pipeline=None,
    scheduler=None,
    lora=None,
    controlnet=None,
):
    # ... function implementation ...
```
● This function, load_pipe, is responsible for loading and configuring the text-to-image generation pipeline. It handles various components like custom pipelines, schedulers, LoRA, and ControlNet. The function also manages model loading, potential quantization, and device placement.
### Inference Time and Throughput Calculation:
```
def calculate_inference_time_and_throughput(height, width, n_steps, model):
    start_time = time.time()
    model(prompt=args.prompt, height=height, width=width, num_inference_steps=n_steps)
    end_time = time.time()
    inference_time = end_time - start_time
    throughput = n_steps / inference_time
    return inference_time, throughput
```
● This function measures the performance of the text-to-image generation process. It calculates both the inference time and throughput (steps per second) for a single run of the model.
### Keyword Argument Handling:
```
def get_kwarg_inputs():
    kwarg_inputs = dict(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=height,
        width=width,
        # ... other keyword arguments ...
    )
    # ... additional argument handling ...
    return kwarg_inputs
```
● The get_kwarg_inputs function prepares a dictionary of keyword arguments for the pipeline. It includes various generation parameters and handles optional arguments like input images and deep caching options.
### Performance Profiling:
```
class IterationProfiler:
    def __init__(self):
        self.begin = None
        self.end = None
        self.num_iterations = 0

    # ... (methods for profiling)
```
● The IterationProfiler class is used for detailed performance profiling of the generation process. It tracks the timing of individual iterations using CUDA events.
### Main Method Details:
#### Pipeline Loading:
```
pipe = load_pipe(
    pipeline_cls,
    args.model,
    variant=args.variant,
    custom_pipeline=args.custom_pipeline,
    scheduler=args.scheduler,
    lora=args.lora,
    controlnet=args.controlnet,
)
```
● This code loads the specified diffusion model pipeline with various customization options.
#### Image Size Determination:
```
pythonCopyheight = args.height or core_net.config.sample_size * pipe.vae_scale_factor
width = args.width or core_net.config.sample_size * pipe.vae_scale_factor
```
● Sets the output image dimensions based on user arguments or model defaults.
#### Compiler Optimization:
```
if args.compiler == "none":
    pass
elif args.compiler == "oneflow":
    pipe = compile_pipe(pipe)
elif args.compiler == "nexfort":
    # ... (nexfort compilation logic)
elif args.compiler in ("compile", "compile-max-autotune"):
    # ... (torch.compile logic)
```
● Applies compiler optimizations to the pipeline based on the specified compiler option.
#### Input Image Handling:
```
if args.input_image is None:
    input_image = None
else:
    input_image = load_image(args.input_image)
    input_image = input_image.resize((width, height), Image.LANCZOS)
```
● Loads and resizes an input image if specified for image-to-image tasks.
#### Control Image Handling:
```
if args.control_image is None:
    if args.controlnet is None:
        control_image = None
    else:
        # ... (create a default control image)
else:
    control_image = load_image(args.control_image)
    control_image = control_image.resize((width, height), Image.LANCZOS)
```
● Prepares a control image for ControlNet, either loading a specified image or creating a default one.
#### Warm-up Runs:
```
if args.warmups > 0:
    # ... (perform warm-up runs)
```
● Executes warm-up runs to trigger compilation and initial optimizations.
#### Main Inference:
```
kwarg_inputs = get_kwarg_inputs()
iter_profiler = IterationProfiler()
# ... (set up profiling callback)
begin = time.time()
output_images = pipe(**kwarg_inputs).images
end = time.time()
● Performs the main image generation inference with profiling.
Performance Reporting:
pythonCopyprint(f"Inference time: {end - begin:.3f}s")
iter_per_sec = iter_profiler.get_iter_per_sec()
if iter_per_sec is not None:
    print(f"Iterations per second: {iter_per_sec:.3f}")
# ... (memory usage reporting)
```
● Reports various performance metrics including inference time and iterations per second.
#### Output Image Saving:
```
if args.output_image is not None:
    output_images[0].save(args.output_image)
```
● Saves the generated image if an output path is specified.
#### Multi-resolution Testing:
```
if args.run_multiple_resolutions:
    # ... (run inference at multiple resolutions)
```
● Tests the model's performance across various image resolutions.
#### Throughput Analysis:
```
if args.throughput:
    steps_range = range(1, 100, 1)
    data, coefficients = generate_data_and_fit_model(pipe, steps_range)
    plot_data_and_model(data, coefficients)
```
● Performs a detailed throughput analysis across different numbers of inference steps and plots the results.
#### Main Execution Block:
```
if __name__ == "__main__":
    main()
```
● Ensures that the main() function is called only when the script is run directly, not when it's imported as a module.
	
</details>


## Results 

<details>
	<summary>Compiler Oneflow</summary>
	
### Text-to-Image
#### SG161222/RealVisXL_V4.0
##### 1024x1024 
###### with OneDiff
	Warmup time: 68.700s
	=======================================
	=======================================
	Inference time: 0.874s
	Iterations per second: 16.183
	Max used CUDA memory : 13.244GiB
	=======================================
 Image:
 ![1024_Oneflow_SG161222_RealVisXL_V4 0](https://github.com/user-attachments/assets/091fe8e3-3506-4826-83cd-49ea78905392)


###### without OneDiff
	Warmup time: 2.439s
	=======================================
	=======================================
	Inference time: 1.521s
	Iterations per second: 8.326
	Max used CUDA memory : 10.465GiB
	=======================================
  Image:
  ![1024_without_Oneflow_SG161222_RealVisXL_V4 0](https://github.com/user-attachments/assets/345a9458-8aba-4368-977c-f40130fa8a12)
##### 512x512 
###### with OneDiff

	Warmup time: 67.218s
	=======================================
	=======================================
	Inference time: 0.332s
	Iterations per second: 44.523
	Max used CUDA memory : 10.031GiB
	=======================================
 Image:
 ![512_Oneflow_SG161222_RealVisXL_V4 0](https://github.com/user-attachments/assets/017e38ec-5d14-4c08-83b9-23fa953c9ba2)

###### without OneDiff
	Warmup time: 1.755s
	=======================================
	=======================================
	Inference time: 0.874s
	Iterations per second: 13.381
	Max used CUDA memory : 7.661GiB
	=======================================

Image:
![512_without_Oneflow_SG161222_RealVisXL_V4 0](https://github.com/user-attachments/assets/959761af-1e87-4104-ada2-9e27568bd8d6)

#### SG161222/RealVisXL_V4.0_Lightning
##### 1024x1024 
###### with OneDiff
	Warmup time: 71.707s
	=======================================
	=======================================
	Inference time: 0.863s
	Iterations per second: 16.257
	Max used CUDA memory : 13.248GiB
	=======================================
 Image:
 ![1024_Oneflow_SG161222_RealVisXL_V4 0_Lightning](https://github.com/user-attachments/assets/94897ae8-32d0-4f84-9a1f-d54bd5984ea6)


###### without OneDiff
	Warmup time: 2.405s
	=======================================
	=======================================
	Inference time: 1.536s
	Iterations per second: 8.325
	Max used CUDA memory : 10.470GiB
	=======================================
 Image:
 ![1024_without_Oneflow_SG161222_RealVisXL_V4 0_Lightning](https://github.com/user-attachments/assets/5fada763-1bf8-47ee-a5ec-54c084a588ea)

 

##### 512x 512 
###### with OneDiff

	Warmup time: 67.914s
	=======================================
	=======================================
	Inference time: 0.337s
	Iterations per second: 42.992
	Max used CUDA memory : 10.085GiB
	=======================================
 
Image:
![512_Oneflow_SG161222_RealVisXL_V4 0_Lightning](https://github.com/user-attachments/assets/8f3ebb72-234f-491c-8362-1c0f169eb61a)


###### without OneDiff
	Warmup time: 1.817s
	=======================================
	=======================================
	Inference time: 0.890s
	Iterations per second: 13.250
	Max used CUDA memory : 7.656GiB
	=======================================
 Image:
 ![512_without_Oneflow_SG161222_RealVisXL_V4 0_Lightning](https://github.com/user-attachments/assets/5ddd67a3-e6b0-49bf-aae0-56ac5173a9c8)
 
### Image-to-Image
For 1024 x 1024 size images i have used 1024 sized image generated from RealVisXL_V4.0 model and same for 512 too. prompt ("trun her into a cyborg") 
#### SG161222/RealVisXL_V4.0 
##### 1024x1024 
###### with OneDiff
	Warmup time: 70.647s
	=======================================
	=======================================
	Inference time: 0.871s
	Iterations per second: 16.199
	Max used CUDA memory : 13.302GiB
	=======================================
Image:
![Test_1024_With_Oneflow_I2I_SG161222_RealVisXL_V4 0](https://github.com/user-attachments/assets/cf95511a-25bb-4187-95ff-34c9257c8e7b)

###### without OneDiff
	Warmup time: 2.313s
	=======================================
	=======================================
	Inference time: 1.522s
	Iterations per second: 8.290
	Max used CUDA memory : 10.471GiB
	=======================================
Image:
![Test_1024_Without_Oneflow_I2I_SG161222_RealVisXL_V4 0](https://github.com/user-attachments/assets/30169f39-12ff-47c3-9c1c-71012eadb89a)

##### 512x512 
###### with OneDiff
	Warmup time: 72.229s
	=======================================
	=======================================
	Inference time: 0.325s
	Iterations per second: 47.863
	Max used CUDA memory : 10.031GiB
	=======================================
 Image:
 ![Test_512_With_Oneflow_I2I_SG161222_RealVisXL_V4 0](https://github.com/user-attachments/assets/6dfa5ae7-c455-4679-81a2-81d894a5aaa3)


###### without OneDiff
	Warmup time: 1.784s
	=======================================
	=======================================
	Inference time: 0.898s
	Iterations per second: 12.942
	Max used CUDA memory : 7.661GiB
	=======================================
 Image:
 ![Test_512_Without_Oneflow_I2I_SG161222_RealVisXL_V4 0](https://github.com/user-attachments/assets/058a5d1e-b711-476e-9302-5040b86cdcbf)


#### timbrooks/instruct-pix2pix
##### 1024x1024 
###### with OneDiff
	Warmup time: 45.665s
	=======================================
	=======================================
	Inference time: 0.888s
	Iterations per second: 13.108
	Max used CUDA memory : 13.079GiB
	=======================================
Image:
![Test_1024_With_Oneflow_I2I_timbrooks_instruct-pix2pix](https://github.com/user-attachments/assets/110adee8-6e4d-40f4-b13a-666264c8b039)
###### without OneDiff
	Warmup time: 2.343s
	=======================================
	=======================================
	Inference time: 1.723s
	Iterations per second: 7.033
	Max used CUDA memory : 4.400GiB
	=======================================
 Image:
 ![Test_1024_Without_Oneflow_I2I_timbrooks_instruct-pix2pix](https://github.com/user-attachments/assets/2b7b00e1-face-422e-acc5-bd48c7697dda)
##### 512 x 512 
###### with OneDiff
	Warmup time: 38.675s
	=======================================
	=======================================
	Inference time: 0.187s
	Iterations per second: 69.570
	Max used CUDA memory : 4.636GiB
	=======================================
 Image:
 ![Test_512_With_Oneflow_I2I_timbrooks_instruct-pix2pix](https://github.com/user-attachments/assets/5a7e4396-db5b-4102-94b9-5c6b43882aaa)


###### without OneDiff
	Warmup time: 1.231s
	=======================================
	=======================================
	Inference time: 0.397s
	Iterations per second: 31.571
	Max used CUDA memory : 2.613GiB
	=======================================
Image:
![Test_512_Without_Oneflow_I2I_timbrooks_instruct-pix2pix](https://github.com/user-attachments/assets/546fda0b-050c-4bcb-b46c-7327ee2f933a)

</details>
<details>
	<summary>Compiler Nexfort</summary>
	
### Text-to-Image
#### SG161222/RealVisXL_V4.0_Lightning	
##### 1024x1024 
###### with OneDiff
	Warmup time: 924.378s
	=======================================
	=======================================
	Inference time: 0.979s
	Iterations per second: 13.871
	Max used CUDA memory : 11.464GiB
	=======================================
  Image:
  ![RealVisXL_withonediff_1024](https://github.com/user-attachments/assets/f58ac114-d1e6-43a0-912c-b759fd65836c)

######	Without OneDiff
  	Warmup time: 2.391s
	=======================================
	=======================================
	Inference time: 1.515s
	Iterations per second: 8.331
	Max used CUDA memory : 10.471GiB
	=======================================
  Image:
  ![RealVisXL_withoutonediff_1024](https://github.com/user-attachments/assets/be2e6f11-4fd7-4441-95f2-cb3dee21dbb8)

##### 512 x 512 
###### with onediff
	Warmup time: 890.209s
	=======================================
	=======================================
	Inference time: 0.704s
	Iterations per second: 17.770
	Max used CUDA memory : 8.956GiB
	=======================================
 Image:
 ![RealVisXL_withonediff_512](https://github.com/user-attachments/assets/2f8b6f0b-8a0e-432d-8efb-c1aff366c2b1)

###### without onediff
	Warmup time: 1.696s
	=======================================
	=======================================
	Inference time: 0.889s
	Iterations per second: 13.081
	Max used CUDA memory : 7.657GiB
	=======================================
 Image:
 ![RealVisXL_withoutonediff_512](https://github.com/user-attachments/assets/d77e3ab0-d447-41ec-946a-b19cb343443d)

#### SG161222/RealVisXL_V4.0
##### 1024x1024 
###### with OneDiff
	Warmup time: 813.568s
	=======================================
	=======================================
	Inference time: 0.976s
	Iterations per second: 13.891
	Max used CUDA memory : 11.465GiB
	======================================
Image:
![RealVisXL4 0_withonediff_1024](https://github.com/user-attachments/assets/8cceb554-d0a0-43e9-919b-bec75552c644)

###### without onediff
       Warmup time: 3.034s
	=======================================
	=======================================
	Inference time: 1.518s
	Iterations per second: 8.333
	Max used CUDA memory : 10.473GiB
	=======================================
 Image:
 ![RealVisXL4 0__withoutonedifftest_1024](https://github.com/user-attachments/assets/66a55a66-6446-48cd-bd7f-8f9f7b17e604)
##### 512 x 512
###### with OneDiff
	Warmup time: 802.404s
	=======================================
	=======================================
	Inference time: 0.697s
	Iterations per second: 17.522
	Max used CUDA memory : 8.956GiB
	=======================================
 Image:
 ![RealVisXL4 0_withonediff_512](https://github.com/user-attachments/assets/de70ec25-84e3-408a-8ce8-e90cdce68a0f)
 
 ###### without OneDiff
 	Warmup time: 1.577s
	=======================================
	=======================================
	Inference time: 0.868s
	Iterations per second: 13.497
	Max used CUDA memory : 7.657GiB
	=======================================
Image:
![RealVisXL4 0_withonediff_512](https://github.com/user-attachments/assets/60d47acf-b961-4f26-9cf9-9d3bd705875d)


### Image-to-image
    for Image 2 Image the images i am using are also the generated images from the above models under different size

#### timbrooks/instruct-pix2pix
##### 1024x1024 Prompt(turn it into a painting) , Image used 1024 x 1024 without onediff generated image by RealVisXL_4.0
###### with OneDiff
	Warmup time: 414.009s
	=======================================
	=======================================
	Inference time: 2.558s
	Iterations per second: 12.674
	Max used CUDA memory : 3.643GiB
	=======================================
 Image:
 ![i2i_1024__timebrooks_withonediff](https://github.com/user-attachments/assets/a99dfb0a-faa3-4a37-85b1-ea0b3d55364e)

###### without OneDiff
	Warmup time: 5.245s
	=======================================
	=======================================
	Inference time: 4.569s
	Iterations per second: 7.035
	Max used CUDA memory : 4.400GiB
	=======================================
 Image:
 ![i2i_1024__timebrooks_withoutonediff](https://github.com/user-attachments/assets/e362e4ac-eb0a-4b58-87df-e835a6735386)

##### 512 X 512  Prompt(turn it into a painting) , Image used 512 x 512 without onediff generated image by RealVisXL_4.0
###### with OneDiff
	Warmup time: 470.570s
	=======================================
	=======================================
	Inference time: 0.790s
	Iterations per second: 42.596
	Max used CUDA memory : 2.693GiB
	=======================================
 Image:
 ![i2i_512_timebrooks_withonediff](https://github.com/user-attachments/assets/be5f35c0-e497-4452-94e8-f922dfb835eb)
###### without OneDiff
	Warmup time: 1.883s
	=======================================
	=======================================
	Inference time: 1.045s
	Iterations per second: 31.124
	Max used CUDA memory : 2.625GiB
Image:
![i2i_512_timebrooks_withoutonediff](https://github.com/user-attachments/assets/1b125d7e-ad53-478c-bcb6-1e5782cd4362)

#### SG161222/RealVisXL_V4.0
##### 1024x1024  Prompt(make it into a cyborg) , Image used 1024 x 1024 without onediff generated image by RealVisXL_4.0
###### with OneDiff 
	Warmup time: 218.133s
	=======================================
	=======================================
	Inference time: 1.079s
	Iterations per second: 15.162
	Max used CUDA memory : 11.489GiB
 	=================================
  Image:
======![i2i_1024_withonediff](https://github.com/user-attachments/assets/07edc615-9123-4827-874f-4134ed98875a)


###### without OneDiff
	Warmup time: 2.355s
	=======================================
	=======================================
	Inference time: 1.585s
	Iterations per second: 8.327
	Max used CUDA memory : 10.474GiB
	=======================================
Image:
![i2i_1024_withoutonediff](https://github.com/user-attachments/assets/f1cdc374-c2b3-49c6-bb1d-263426b99464)


##### 512x512  Prompt(turn it into a painting) , Image used 512 x 512 without onediff generated image by RealVisXL_4.0
###### with OneDiff
	Warmup time: 553.390s
	=======================================
	=======================================
	Inference time: 0.542s
	Iterations per second: 25.336
	Max used CUDA memory : 8.977GiB
	=======================================
 Image:
 ![testi2i one diff ](https://github.com/user-attachments/assets/7a4d6ad7-0b7a-42fd-b267-b394ef4c081b)
 
 ###### without OneDiff
 	Warmup time: 1.654s
	=======================================
	=======================================
	Inference time: 0.831s
	Iterations per second: 13.513
	Max used CUDA memory : 7.655GiB
	=======================================
 Image:
 ![i2i_512_withoutonediff](https://github.com/user-attachments/assets/a9273116-ab48-422e-a01a-76c73a4ca8af)
 </details>

## Conclusion
it is evident that OneDiff consistently reduced inference time, leading to increased inference speed.While the source does not explicitly analyze the impact of OneDiff on image quality, it does provide sample images generated with and without OneDiff for each model and resolution.  A subjective visual comparison of these images suggests that the image quality remains largely unaffected by OneDiff optimization. 

 ## Persnol Suggestion
Working with OneDiff in the current scenario has proven challenging due to several factors. The lack of comprehensive documentation has made it difficult to navigate and implement the library effectively. Testing capabilities are limited, as OneDiff doesn't currently support Kaggle P100 and T4 GPUs, which restricts the environments in which it can be evaluated. Furthermore, the library ecosystem surrounding OneDiff appears to be fragile, with interdependencies causing conflicts. Installing one component often leads to unexpected issues with others, creating a cascade of compatibility problems. This instability in the development environment has made it cumbersome to set up and maintain a reliable testing framework for OneDiff.
