import torch

# Shared Defaults
DEFAULT_VARIANT = None
DEFAULT_CUSTOM_PIPELINE = None
DEFAULT_SCHEDULER = "EulerAncestralDiscreteScheduler"
DEFAULT_LORA = None
DEFAULT_CONTROLNET = None
DEFAULT_STEPS = 30
DEFAULT_NEGATIVE_PROMPT = ""
DEFAULT_SEED = 333
DEFAULT_WARMUPS = 1
DEFAULT_BATCH = 1
DEFAULT_CONTROL_IMAGE = None
DEFAULT_OUTPUT_IMAGE = None
DEFAULT_EXTRA_CALL_KWARGS = None
DEFAULT_CACHE_INTERVAL = 3
DEFAULT_CACHE_LAYER_ID = 0
DEFAULT_CACHE_BLOCK_ID = 0
DEFAULT_COMPILER = "nexfort"
DEFAULT_COMPILER_CONFIG = None
DEFAULT_QUANTIZE_CONFIG = None
DEFAULT_DTYPE = torch.float16
DEFAULT_DEVICE = "cuda"

# Text-to-Image Specific Defaults
DEFAULT_T2I_MODEL = "SG161222/RealVisXL_V4.0"
DEFAULT_T2I_PROMPT = "best quality, realistic, unreal engine, 4K, a cat sitting on human lap"
DEFAULT_T2I_HEIGHT = None # Will be derived from model
DEFAULT_T2I_WIDTH = None  # Will be derived from model
DEFAULT_T2I_INPUT_IMAGE = None # T2I does not use input image by default

# Image-to-Image (InstructPix2Pix) Specific Defaults
DEFAULT_I2I_MODEL = "timbrooks/instruct-pix2pix"
# DEFAULT_I2I_INPUT_IMAGE is REQUIRED, so no default here. Set via command line.
# Example value if needed for testing: "https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg"
DEFAULT_I2I_PROMPT = "make it into a painting"
DEFAULT_I2I_HEIGHT = 512
DEFAULT_I2I_WIDTH = 512
