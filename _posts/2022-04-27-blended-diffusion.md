# Blended diffusion model
Contribution:
1.  propose the first solution for general-purpose region-based image editing, using natural language guidance, applicable to real, diverse images.
2.  background preservation technique guarantees that unaltered regions are perfectly preserved.
3.  a simple augmentation technique significantly reduces the risk of adversarial results, allowing to use gradient-based diffusion guidance.
---

Before going deeper, two pre-requisite model are essential to understand the overall idea, which are Contrastive Language-Image Pre-training (**CLIP**) and Denoising Diffusion Probababilistic Model (**DDPM**). CLIP is a transformer model that has a very strong visual language ability through the pre-training from 400 millions of image-text pair by learning their joint representations. On the other hand, DDPM is another class of generative models which repeatedly applying a small Gaussian Noise into the input image $x_0$ into almost an isotopic Gaussian noise $x_t$, and a parameterized neural network will learn to denoise $x_t$. 
![](https://i.imgur.com/qjO8tbm.png)

### Method
#### First experiment algorithm (**Local Clip-guided diffusion**)

![](https://i.imgur.com/kJpYli3.png)
Quite a naive approach to combine CLIP model and DDPM to perform the text-guided image editing task. A pretrained CLIP model will act as a classifier to guide the DDPM to steer the generated output to match the prompt as much as possible. In short, the diffusion model will generate the output based CLIP loss, which is the cosine distance between generated output embedding and text prompt embedding as shown below:

$$D_{CLIP}(x,d,m) = D_c(CLIP_{img}(x \odot m),CLIP_{txt}(d))$$

where $x$ is input image, $d$ is the text prompt, $m$ is the binary mask of ROI, $D_c$ is the cosine distance and $\odot$ is the element-wise multiplication. The CLIP loss is only consider on the local (generated output under the mask). Thus, another loss is required as we want the surrounding region (region other than masked) to be the same as input image. A background preservation loss $D_{bg}$ is introduced for this purpose.

$$D_{bg} (x_1,x_2) = \frac{1}{2}(MSE(x_1,x_2)+ LPIPS(x_1,x_2)) $$

where $MSE$ is the $L_2$ norm of the pixel-wise difference between the images, and LPIPS is the Learned Perceptual Image Patch Similarity metric. 

The overall loss now become weighted sum of $D_{CLIP}+ \lambda D_{bg}$, where $\lambda$ is the hyperparameter where $\lambda = 0$ represent no background constraint. However, this approach will suffer from issue above where, the surrounding region does not preserved according to input and there is a trade-off between foreground (mask region that we wish to replace object) and background (surrounding region). If we wish to preserve the background, the foreground output is fairly limited as shown in last image ($\lambda = 10000$), while less $\lambda$ show inability in preserving the background.


#### Proposed Method (**Text-driven blended diffusion**)
![](https://i.imgur.com/a54p4Nl.png)
This method can be summarized in figure above, where we will consume three input, text prompt $d$, mask $m$ and input image $x$ similar to first algorithm. The difference is that, the input image will first be transformed into a pure Gaussian Noise image through the forward diffusion process, which can be summarized by using reparameterization trick:
$$ x_{t}=\sqrt{\bar{\alpha}_{t}} x_{0}+\sqrt{1-\bar{\alpha}_{t}} \epsilon$$

where $\bar{\alpha}_{t}$ =  $\prod_{s=0}^{t} \alpha_{s}$ and $\alpha_{t}$ = $1-\beta_{t}$  and $\epsilon \sim \mathcal{N}(0, \mathbf{I})$.

> Any $x_t$ from $t$ step is easy to sample using the equation above.

$x_t$ will then denoised by a parameterized U-Net to generate desire output $x_{t-1,fg}$ according to prompt and CLIP guidance . In addition, a noised version of the background $x_{t−1,bg}$ is obtained from the input image using the same equation above. Both of the foreground output and background will be blend using:

$$x_{t-1}=x_{t-1, f g} \odot m+x_{t-1, b g} \odot(1-m)$$

The process will be repeated according to $t$ steps. The gradients of CLIP loss w.r.t transformed outputs will be help to steer the generation towards desired output. Finally, the entire region outside the mask is replace with the corresponding region from input image in the final step. The main hypothesis is that if the first denoising step produce a non-coherent result, the later step can help to recover the coherence.


**Extending augmentation**
The objective of applying extending augmentations is to avoid adversarial attacks (generate images that reduce the CLIP loss without produce the desired result). At each diffusion steps, several augmentations is applied to the intermediate and the gradient of CLIP loss that are used to guide the diffusion is computed by averaging the gradients w.r.t each projectively transformed copies.



### Implementation Details
- DDPM of resolution 256 × 256
- 100 sampling steps
- Diffusion step = 75
- First resize input image to 224 x 224 to match CLIP size, next create $n$ copies of this image and perform a different random projective transformation on each copy, along with the same transformation on the corresponding mask. Finally, calculate the gradients using the CLIP loss w.r.t each one of the transformed copies and average all the gradients
- Number of extending augmentations = 16
- Number of total repetitions = 64 (How many variations to generate and rank them using CLIP)
- Two pre-trained model, CLIP (ViT-B/16) and unconditional DDPM
- Loss = normal MSE loss








