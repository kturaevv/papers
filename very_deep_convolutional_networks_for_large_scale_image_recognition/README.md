### Very Deep Convolutional Networks For Large-Scale Image Recognition
---
This paper highlights the impact of network depth on achieving state-of-the-art (SOTA) results in image classification by extending convolutional networks to 19 layers, achieved through consistent use of small 3x3 kernels. Key contributions include:

- showing the importance of network depth in enhancing performance.
- showing that local response normalization had minimal impact on accuracy.
- showing that stacking small 3x3 convolutional kernels achieves the same receptive field as larger filters while being more efficient.
- showing that smaller stacked kernels significantly outperform larger filters in terms of accuracy (by roughly 80%).
- showing that spatial granularity matters, as networks with 3x3 kernels outperformed those with 1x1 kernels at equivalent depth.
- showing the effectiveness of scale jittering for better generalization.
- introducing weight initialization by using smaller pre-trained model
- successfully training without any normalization


```python
import torch
from torch import nn

class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        
        self.image_features = nn.Sequential(
            # Stack 1
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Stack 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Stack 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Stack 4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Stack 5
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classification = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
            nn.Softmax(dim = 1),
        )
    
    def forward(self, x):
        x = self.image_features(x)
        x = x.view(x.size(0), -1)
        x = self.classification(x)
        return x

x = torch.randn((1, 3, 224, 224))    
model = VGG16()
output = model(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")
```

    Input shape: torch.Size([1, 3, 224, 224])
    Output shape: torch.Size([1, 1000])
    Number of model parameters: 138357544

