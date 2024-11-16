# Implementation-of-filter
## Aim:
To implement filters for smoothing and sharpening the images in the spatial domain.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1: Load the Image.
</br>Read the image file using a suitable image processing library like OpenCV or PIL. Convert the image to grayscale if necessary for grayscale filtering techniques.
</br> 

### Step2:  Choose a Filter.
</br>Decide on the type of filter you want to apply based on your desired outcome. Some common filters include:

a. Averaging filter

b. Gaussian filter

c. Median filter

d. Laplacian filter
</br> 

### Step3: Create the Filter Kernel.
</br>A filter kernel is a small matrix that is applied to each pixel in the image to produce the filtered result. The size and values of the kernel determine the filter's behavior. For example, an averaging filter kernel has all elements equal to 1/N, where N is the kernel size.
</br> 

### Step4: Apply the Filter.
</br>Use the library's functions to apply the filter to the image. The filtering process typically involves convolving the image with the filter kernel.
</br> 

### Step5: Display or Save the Result.
</br>Visualize the filtered image using a suitable method (e.g., OpenCV's imshow, Matplotlib). Save the filtered image to a file if needed.
</br> 

## Program:

</br>

### 1. Smoothing Filters:

i) Using Averaging Filter
```Python
import cv2
import matplotlib.pyplot as plt
import numpy as np
image1=cv2.imread("flower.jpg")
image2=cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)
kernel=np.ones((11,11),np.float32)/169
image3=cv2.filter2D(image2,-1,kernel)
plt.figure(figsize=(9,9))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(image3)
plt.title("Average Filter Image")
plt.axis("off")
plt.show()
```
![WhatsApp Image 2024-11-16 at 10 21 21 PM](https://github.com/user-attachments/assets/81acf1d5-939e-4044-b455-7948e0987ef6)

ii) Using Weighted Averaging Filter:
```Python
kernel1=np.array([[1,2,1],[2,4,2],[1,2,1]])/16
image3=cv2.filter2D(image2,-1,kernel1)
plt.figure(figsize=(9,9))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(image3)
plt.title("Weighted Average Filter Image")
plt.axis("off")
plt.show()

```
![WhatsApp Image 2024-11-16 at 10 21 31 PM](https://github.com/user-attachments/assets/950f1d73-bffd-4537-ac5a-cc9cb69f44bc)

iii) Using Gaussian Filter: 
```Python
gaussian_blur=cv2.GaussianBlur(image2,(33,33),0,0)
plt.figure(figsize=(9,9))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(gaussian_blur)
plt.title("Gaussian Blur")
plt.axis("off")
plt.show()
```
![WhatsApp Image 2024-11-16 at 10 21 36 PM](https://github.com/user-attachments/assets/7aea38d2-b657-4375-b0f3-d1d5f2bc7b3a)

iv)Using Median Filter:
```Python
median=cv2.medianBlur(image2,13)
plt.figure(figsize=(9,9))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(median)
plt.title("Median Blur")
plt.axis("off")
plt.show()
```
![WhatsApp Image 2024-11-16 at 10 21 41 PM](https://github.com/user-attachments/assets/2139ae03-c89c-442f-b8c8-69bd91af5c8d)

### 2. Sharpening Filters
i) Using Laplacian Linear Kernal
```Python

kernel2=np.array([[-1,-1,-1],[2,-2,1],[2,1,-1]])
image3=cv2.filter2D(image2,-1,kernel2)
plt.figure(figsize=(9,9))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(image3)
plt.title("Laplacian Kernel")
plt.axis("off")
plt.show()
```
![WhatsApp Image 2024-11-16 at 10 21 50 PM](https://github.com/user-attachments/assets/3e9714b0-3974-4ad7-a44c-f7b16678be87)

ii) Using Laplacian Operator
```Python

laplacian=cv2.Laplacian(image2,cv2.CV_64F)
plt.figure(figsize=(9,9))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(laplacian)
plt.title("Laplacian Operator")
plt.axis("off")
plt.show()

```
![WhatsApp Image 2024-11-16 at 10 21 58 PM](https://github.com/user-attachments/assets/90263177-b5ab-4648-a7fe-d0bef47facf8)

## Result:
Thus the filters are designed for smoothing and sharpening the images in the spatial domain.
