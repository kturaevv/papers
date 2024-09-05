# Going deeper with convolutions
---

This paper introduced new, compute efficient CNN architecture - GoogLeNet. The architecture was inspired by theoretical work from Arora et al. on approximating optimal sparse neural network structures using readily available dense components. The authors developed a practical solution that maintained the benefits of theoretical sparse networks while remaining computationally efficient through clever use of dense CNN layers.

The core innovation was the Inception module - a carefully crafted convolutional block that could be stacked repeatedly while managing computational complexity. This 22-layer deep architecture achieved state-of-the-art results while using 12x fewer parameters than AlexNet (the 2012 winner).

Key observations:

- Usage of Network in Network architecture or multi-scale processing: basically instead of using one CNN layer, authors used a set of kernels each of different size 1x1 3x3 5x5 to capture information at different levels of scale. 
- Strategic use of 1x1 convolutions before larger kernels for dimension reduction, which also incorporate ReLU activation, providing additional non-linearity benefits
- Auxiliary classifiers attached to intermediate layers during training to improve gradient flow and add regularization (removed during inference)
- Extensive data augmentation using variable-sized image patches (8% to 100% of image area)
- Implementation of photometric distortions to reduce overfitting
- Successfully adapted for object detection by combining with R-CNN approach
- Demonstrated significant computational efficiency compared to contemporary architectures
