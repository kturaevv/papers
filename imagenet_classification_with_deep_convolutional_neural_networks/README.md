### ImageNet Classification with Deep Convolutional Neural Networks
---

This seminal paper demonstrated the practical effectiveness of deep Convolutional Neural Networks (CNNs) by winning the 2012 ImageNet challenge (ILSVRC) and sparked a new wave of interest in AI. Most notable was the dramatic reduction in error rates achieved through a deep architecture of 5 convolutional layers and 3 fully connected layers, powered by 2 GPUs that not only shared the load but specialized in learning different features.

They effectively showed how to address key challenges in training deep networks, including:

- usage of ReLU activation function against Tanh which proved to converge 6x faster
- usage of Local Response Normalization (LRN) layer, which was important before batch normalization existed
- usage of two GPUs for training one big model with split architecture
- usage of dropout layers to avoid overfitting on the large dataset
- usage of overlapping pooling layers for improved performance
- usage of computationally free data augmentation techniques

The paper's success transformed computer vision by proving deep CNNs could work in practice on large-scale datasets, setting the foundation for modern deep learning and the trend toward even deeper architectures in following years.


```python
import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        
        self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Layer 2
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2), # split input into 2 GPU groups
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            # Layer 3
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Layer 4
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            # Layer 5
            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.classifier = nn.Sequential(
            # Layer 6
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            # Layer 7
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # Layer 8 (output)
            nn.Linear(4096, num_classes),
        )
        
        self._initialize_weights()
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        """
        Initialize model weights as described in the paper:
        - Conv layers: drawn from N(0, 0.01)
        - Bias terms: initialized with 1 for conv layers 2, 4, 5
        - Bias terms: initialized with 0 for remaining layers
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    if m in [self.features[4], self.features[10], self.features[12]]:  # layers 2, 4, 5
                        nn.init.constant_(m.bias, 1)
                    else:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)

model = AlexNet(num_classes=1000)
x = torch.randn(1, 3, 227, 227)
output = model(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")
```

    Input shape: torch.Size([1, 3, 227, 227])
    Output shape: torch.Size([1, 1000])
    Number of model parameters: 60965224

