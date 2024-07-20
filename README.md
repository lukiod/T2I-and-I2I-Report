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
##### 1024x1024 
###### with OneDiff

#### SG161222/RealVisXL_V4.0
##### 1024x1024 
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

##### 512x512 
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

## Conclusion

Did it increase the inference speed while image quality was not affected?

