# Deep Generative Models course, MIPT + YSDA, 2024

## Description
The course is devoted to modern generative models (mostly in the application to computer vision).

We will study the following types of generative models:
- autoregressive models,
- latent variable models,
- normalization flow models,
- adversarial models,
- diffusion and score models.

Special attention is paid to the properties of various classes of generative models, their interrelationships, theoretical prerequisites and methods of quality assessment.

The aim of the course is to introduce the student to widely used advanced methods of deep learning.

The course is accompanied by practical tasks that allow you to understand the principles of the considered models.

## Contact the author to join the course or for any other questions :)

- **telegram:** [@roman_isachenko](https://t.me/roman_isachenko)
- **e-mail:** roman.isachenko@phystech.edu

## Materials

| # | Date | Description | Slides |
|---|---|---|---|
| 1 | September, 10 | <b>Lecture 1:</b> Logistics. Generative models overview and motivation. Problem statement. Divergence minimization framework. Autoregressive models (PixelCNN). | [slides](lectures/lecture1/Lecture1.pdf) |
|  |  | <b>Seminar 1:</b> Introduction. Maximum likelihood estimation. Histograms. Bayes theorem. PixelCNN | [slides](seminars/seminar1/seminar1.ipynb) |
| 2 | September, 17 | <b>Lecture 2:</b> Normalizing Flow (NF) intuition and definition. Linear NF. Gaussian autoregressive NF. Coupling layer (RealNVP). | [slides](lectures/lecture2/Lecture2.pdf) |
|  |  | <b>Seminar 2:</b> Planar and Radial Flows. Forward vs Reverse KL. | [slides](seminars/seminar2/seminar2.ipynb) |
| 3 | September, 24 | <b>Lecture 3:</b> Forward and reverse KL divergence for NF. Latent variable models (LVM). Variational lower bound (ELBO). EM-algorithm. | [slides](lectures/lecture3/Lecture3.pdf) |
|  |  | <b>Seminar 3:</b> Forward vs Reverse KL. RealNVP | [slides](seminars/seminar3/seminar3.ipynb) |
| 4 | October, 1 | <b>Lecture 4:</b> Amortized inference, ELBO gradients, reparametrization trick. Variational Autoencoder (VAE). NF as VAE model. Discrete VAE latent representations. | [slides](lectures/lecture4/Lecture4.pdf) |
|  |  | <b>Seminar 4:</b> Gaussian Mixture Model (GMM). GMM and MLE. ELBO and EM-algorithm. GMM via EM-algorithm. Variational EM algorithm for GMM. | [slides](seminars/seminar4/seminar4.ipynb) |
| 5 | October, 8 | <b>Lecture 5:</b> Vector quantization, straight-through gradient estimation (VQ-VAE). Gumbel-softmax trick (DALL-E). ELBO surgery and optimal VAE prior. Learnable VAE prior. | [slides](lectures/lecture5/Lecture5.pdf) |
|  |  | <b>Seminar 5:</b> VAE: Implementation hints. Vanilla 2D VAE coding. VAE on Binarized MNIST visualization. Posterior collapse. Beta VAE on MNIST.| [slides](seminars/seminar5/seminar5.ipynb) |
| 6 | October, 15 | <b>Lecture 6:</b> Likelihood-free learning. GAN optimality theorem. Wasserstein distance. | [slides](lectures/lecture6/Lecture6.pdf) |
|  |  | <b>Seminar 6:</b>  KL vs JS divergences. Vanilla GAN in 1D coding. Mode collapse and vanishing gradients. Non-saturating GAN. | [slides](seminars/seminar6/seminar6.ipynb) |
| 7 | October, 22 | <b>Lecture 7:</b> Wasserstein GAN (WGAN). f-divergence minimization. GAN evaluation (FID, Precision-Recall, truncation trick). | [slides](lectures/lecture7/Lecture7.pdf) |
|  |  | <b>Seminar 7:</b> WGAN and WGAN-GP. | [slides](seminars/seminar7/seminar7.ipynb) |
| 8 | October, 29 | <b>Lecture 8:</b>  Langevin dynamic. Score matching (Denoising score matching, Noise Conditioned Score Network (NCSN)). Forward gaussian diffusion process. | [slides](lectures/lecture8/Lecture8.pdf) |
|  |  | <b>Seminar 8:</b> StyleGAN. | [slides](seminars/seminar8/StyleGAN.ipynb) |
| 9 | November, 5 | <b>Lecture 9:</b> Denoising score matching for diffusion. Reverse Gaussian diffusion process. Gaussian diffusion model as VAE. ELBO for DDPM. | [slides](lectures/lecture9/Lecture9.pdf) |
|  |  | <b>Seminar 9:</b> Noise Conditioned Score Network (NCSN). | [slides](seminars/seminar9/seminar9.ipynb) |
| 10 | November, 12 | <b>Lecture 10:</b> Denoising diffusion probabilistic model (DDPM): reparametrization and overview. Denoising diffusion as score-based generative model. Model guidance: classifier guidance, classfier-free guidance. | [slides](lectures/lecture10/Lecture10.pdf) |
|  |  | <b>Seminar 10:</b>  Denoising diffusion probabilistic model (DDPM). Denoising Diffusion Implicit Models (DDIM).| [slides](seminars/seminar10/seminar10.ipynb) |
| 11 | November, 19 | <b>Lecture 11:</b> Continuous-in-time NF and neural ODE. Continuity equation for NF log-likelihood. FFJORD and Hutchinson's trace estimator. Adjoint method for continuous-in-time NF.  | [slides](lectures/lecture11/Lecture11.pdf) |
|  |  | <b>Seminar 11:</b> Guidance. CLIP, GLIDE, DALL-E 2, Imagen, Latent Diffusion Model. | [slides](seminars/seminar11/seminar11.ipynb) |
| 12 | November, 26 | <b>Lecture 12:</b> SDE basics. Kolmogorov-Fokker-Planck equation. Probability flow ODE. Reverse SDE. Variance Preserving and Variance Exploding SDEs. | [slides](lectures/lecture12/Lecture12.pdf) |
|  |  | <b>Seminar 12:</b>  Latent Diffusion Models. Recap and colab playground.| [slides](seminars/seminar12/seminar12.ipynb) <a href="https://colab.research.google.com/github/r-isachenko/2024-DGM-MIPT-YSDA-course/blob/main/seminars/seminar12/seminar12_SD.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| 13 | December, 3 | <b>Lecture 13:</b> Score-based generative models through SDE. Flow matching. Conditional flow matching. Conical gaussian paths. | [slides](lectures/lecture13/Lecture13.pdf) |
|  |  | <b>Seminar 13:</b> Latent Diffusion Models. Code. | [slides](seminars/seminar13/seminar13_SD.ipynb) <a href="https://colab.research.google.com/github/r-isachenko/2024-DGM-MIPT-YSDA-course/blob/main/seminars/seminar13/seminar13_SD.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
<!---
| 13 | December, 10 | <b>Lecture 14:</b> Link with diffusion and score matching. | [slides](lectures/lecture14/Lecture14.pdf) |
|  |  | <b>Seminar 14:</b>  | [slides](seminars/seminar14/seminar14.ipynb) |
-->

## Homeworks
| Homework | Date | Deadline | Description | Link |
|---------|------|-------------|--------|-------|
| 1 | September, 18 | October, 2 | <ol><li>Theory (Аlpha-divergences, curse of dimensionality, NF expressivity).</li><li>PixelCNN (receptive field, autocomplete) on MNIST.</li><li>ImageGPT on MNIST.</li></ol> | [![Open In Github](https://img.shields.io/static/v1.svg?logo=github&label=Repo&message=Open%20in%20Github&color=lightgrey)](homeworks/hw1.ipynb)<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/r-isachenko/2024-DGM-MIPT-YSDA-course/blob/main/homeworks/hw1.ipynb) |
| 2 | October, 3 | October, 16 | <ol><li>Theory (IWAE theory, Gaussian VAE - I, Gaussian VAE - II).</li><li>RealNVP on MNIST.</li><li>ResNetVAE on CIFAR10 data.</li></ol> | [![Open In Github](https://img.shields.io/static/v1.svg?logo=github&label=Repo&message=Open%20in%20Github&color=lightgrey)](homeworks/hw2.ipynb)<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/r-isachenko/2024-DGM-MIPT-YSDA-course/blob/main/homeworks/hw2.ipynb) |
| 3 | October, 17 | October, 30 | <ol><li>Theory (ELBO surgery, Gumbel-Max trick, Least Squares GAN).</li><li>VQ-VAE on MNIST.</li><li>Wasserstein GANs for CIFAR 10.</li></ol> | [![Open In Github](https://img.shields.io/static/v1.svg?logo=github&label=Repo&message=Open%20in%20Github&color=lightgrey)](homeworks/hw3.ipynb)<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/r-isachenko/2024-DGM-MIPT-YSDA-course/blob/main/homeworks/hw3.ipynb) |
| 4 | October, 31 | November, 14 | <ol><li>Theory (Conjugate functions, FID for Normal distributions, Implicit score matching).</li><li>Denoising score matching on 2D data.</li><li>NCSN on MNIST.</li></ol> | [![Open In Github](https://img.shields.io/static/v1.svg?logo=github&label=Repo&message=Open%20in%20Github&color=lightgrey)](homeworks/hw4.ipynb)<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/r-isachenko/2024-DGM-MIPT-YSDA-course/blob/main/homeworks/hw4.ipynb) |
| 5 | November, 15 | November, 29 | <ol><li>Theory (Gaussian diffusion, Strided sampling, Conditioned reverse distribution for NCSN).</li><li>DDPM on 2D data.</li><li>DDPM on MNIST.</li></ol> | [![Open In Github](https://img.shields.io/static/v1.svg?logo=github&label=Repo&message=Open%20in%20Github&color=lightgrey)](homeworks/hw5.ipynb)<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/r-isachenko/2024-DGM-MIPT-YSDA-course/blob/main/homeworks/hw5.ipynb) |
| 6 | December, 1 | December, 15 | <ol><li>Theory (KFP theorem, DDPM as SDE discretization, Flow matching distribution).</li><li>Continuous-time Normalizing Flows on 2D data.</li><li>Flow matching on MNIST.</li></ol> |  [![Open In Github](https://img.shields.io/static/v1.svg?logo=github&label=Repo&message=Open%20in%20Github&color=lightgrey)](homeworks/hw6.ipynb)<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/r-isachenko/2024-DGM-MIPT-YSDA-course/blob/main/homeworks/hw6.ipynb) |


## Game rules
- 6 homeworks each of 13 points = **78 points**
- oral cozy exam = **26 points**
- maximum points: 78 + 26 = **104 points**
### Final grade: `floor(relu(#points/8 - 2))`

## Prerequisities
- probability theory + statistics
- machine learning + basics of deep learning
- python + pytorch

## Previous episodes
- [2024, spring, AIMasters](https://github.com/r-isachenko/2024-DGM-AIMasters-course)
- [2023, autumn, MIPT](https://github.com/r-isachenko/2023-DGM-MIPT-course)
- [2022-2023, autumn-spring, MIPT](https://github.com/r-isachenko/2022-2023-DGM-MIPT-course)
- [2022, autumn, AIMasters](https://github.com/r-isachenko/2022-2023-DGM-AIMasters-course)
- [2022, spring, OzonMasters](https://github.com/r-isachenko/2022-DGM-Ozon-course)
- [2021, autumn, MIPT](https://github.com/r-isachenko/2021-DGM-MIPT-course)
- [2021, spring, OzonMasters](https://github.com/r-isachenko/2021-DGM-Ozon-course)
- [2020, autumn, MIPT](https://github.com/r-isachenko/2020-DGM-MIPT-course)

