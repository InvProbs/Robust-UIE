# A robust Physics-based Fixd Point Deep Learning Approach for Underwater Image Enhancement

***by Peimeng Guan, Naveed Iqbal, Mark A. Davenport and Mudassir Masood***

This repo contains the code for paper "A robust Physics-based Fixd Point Deep Learning Approach for Underwater Image Enhancement" submitted to IEEE Journal of Oceanic Engineering. 

*************

Underwater image enhancement is crucial for applications such as marine exploration, environmental monitoring, and autonomous navigation. The enhancement process is typically modeled as an inverse problem using an idealized physics-based formulation, which, despite its closed-form inversion, often lacks accuracy. Physics-based methods rely on this formulation, offering inherent robustness through structural priors, but their reconstruction quality is often limited by the model’s restricted expressiveness. In contrast, data-driven approaches, particularly iterative architectures, offer better visual fidelity but are vulnerable to noise and adversarial perturbations under model mismatch. To address this tradeoff, we propose PhyNN-fixpoint, a robust iterative framework that jointly corrects model mismatch while enforcing physics consistency. By reformulating the enhancement as a fixed-point problem, PhyNN-fixpoint leverages both the closed-form inversion and learned model corrections to construct a parameterized update scheme. This integration of physical priors with learned refinements enhances both reconstruction accuracy and robustness. Extensive evaluations on multiple benchmark datasets demonstrate the superior performance and robustness of our approach to adversarial and Gaussian attacks compared to existing methods.

## Environment Setup
```
git clone https://github.com/InvProbs/Robust-UIE.git
cd Robust-UIE
pip install -r requirements.txt
```

## Training
Please refer to `main.py` to start training.
