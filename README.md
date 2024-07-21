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

### How to run the file independently
If you want to run with onediff use compiler nexfort otherwise none
#### Text 2 Image Conversion
`python3 ./text_to_image.py  --scheduler none --steps 50 --height 1024 --width 1024 --compiler none --compiler-config '{"mode": "max-optimize:max-autotune:low-precision", "memory_format": "channels_last", "dynamic": true}' --output-image ./test.png`
#### Image 2 Image Conversion
`python3 testi2i.py --input-image ./RealVisXL_withoutonediff_1024.png --height 1024 --width 1024 --compiler none --output-image ./i2i_1024__timebrooks_withoutonediff.png --prompt "turn it into a painting painted by paintbrush"`
## Results 

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

## Conclusion
it is evident that OneDiff consistently reduced inference time, leading to increased inference speed.While the source does not explicitly analyze the impact of OneDiff on image quality, it does provide sample images generated with and without OneDiff for each model and resolution.  A subjective visual comparison of these images suggests that the image quality remains largely unaffected by OneDiff optimization. 

 ## Persnol Suggestion
Working with OneDiff in the current scenario has proven challenging due to several factors. The lack of comprehensive documentation has made it difficult to navigate and implement the library effectively. Testing capabilities are limited, as OneDiff doesn't currently support Kaggle P100 and T4 GPUs, which restricts the environments in which it can be evaluated. Furthermore, the library ecosystem surrounding OneDiff appears to be fragile, with interdependencies causing conflicts. Installing one component often leads to unexpected issues with others, creating a cascade of compatibility problems. This instability in the development environment has made it cumbersome to set up and maintain a reliable testing framework for OneDiff.
