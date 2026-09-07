import torch
from models.tcformer import TCFormer

def test():
    model = TCFormer(n_channels=22, n_classes=4)
    x = torch.randn(2, 22, 1000)
    outputs = model(x)
    print(f"Number of outputs: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"Output {i} shape: {out.shape}")
        
if __name__ == "__main__":
    test()