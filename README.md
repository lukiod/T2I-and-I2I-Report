# T2I-and-I2I-Report
# What is OneDiff

	OneDiff is an acceleration library for diffusion models, allowing for faster inference with minimal code changes. The name stands 	for "one line of code to accelerate diffusion models". It achieves this through features like PyTorch Module compilation and 	optimised GPU Kernels.


## The Research Question

Does OneDiff improve the inference speed of our T2I and I2I pipelines while still providing high quality.

## Method

How did you test this? Performed 10 inference requests without and with OneDiff and compared both the inference time and image quality.

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
	Withour Onediff- Here are the model Inference time: 9.939s Iterations per second: 1.236 Max used CUDA memory : 9.718GiB 

### Image-to-image

#### timbrooks/instruct-pix2pix

#### SG161222/RealVisXL_V4.0

## Conclusion

Did it increase the inference speed while image quality was not affected?

