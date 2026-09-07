import torch
from models.tcformer import TCFormer
from models.classification_module import ClassificationModule

def test():
    print("Initializing model...")
    model = TCFormer(n_channels=22, n_classes=4)
    print("Initializing ClassificationModule...")
    pl_module = ClassificationModule(model=model, n_classes=4)
    
    print("Creating mock batch...")
    x = torch.randn(2, 22, 1000)
    y = torch.tensor([0, 3])
    batch = (x, y)
    
    print("Testing shared_step (train mode)...")
    loss, acc = pl_module.shared_step(batch, batch_idx=0, mode="train")
    print(f"Loss: {loss.item()}, Acc: {acc.item()}")
    
    print("Testing predict_step...")
    preds = pl_module.predict_step(batch, batch_idx=0)
    if isinstance(preds, dict):
        print("Prediction keys:", preds.keys())
        for k, v in preds.items():
            print(f"  {k} shape: {v.shape}")
    else:
        print("Prediction shape:", preds.shape)
        
    print("All tests passed!")

if __name__ == "__main__":
    test()