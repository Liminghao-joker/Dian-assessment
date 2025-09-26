# test_resnet18.py
import torch
from Basic_Resnet18 import ResNet18

# 小样本过拟合函数
def overfit_small_sample(model, device, num_classes: int):
    torch.manual_seed(42)
    model.train()

    n_samples = 8
    inputs = torch.randn(n_samples, 3, 224, 224, device=device)
    targets = torch.randint(0, num_classes, (n_samples,), device=device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.0)

    max_epochs = 300
    stop_threshold = 0.01

    for epoch in range(1, max_epochs + 1):
        permutation = torch.randperm(n_samples, device=device)
        epoch_loss = 0.0
        for idx in permutation.split(4):  # batch size=4
            batch_x = inputs[idx]
            batch_y = targets[idx]
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        avg_loss = epoch_loss / n_samples
        if epoch % 20 == 0 or avg_loss < stop_threshold:
            print(f"[overfit] epoch={epoch:03d} avg_loss={avg_loss:.6f}")
        if avg_loss < stop_threshold:
            print("达到阈值，提前停止。")
            break

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 10

    model = ResNet18(num_classes=num_classes).to(device)

    # 前向检查
    x = torch.randn(4, 3, 224, 224, device=device)
    y = model(x)
    assert y.shape == (4, num_classes), f"unexpected output shape: {y.shape}"
    print("forward ok:", y.shape)

    # 反向检查
    target = torch.randint(0, num_classes, (4,), device=device)
    loss = torch.nn.CrossEntropyLoss()(y, target)
    loss.backward()
    print("backward ok, loss=", float(loss))

    # 参数量统计
    params = sum(p.numel() for p in model.parameters())
    print("params:", params)

    # 做小样本拟合
    overfit_small_sample(model, device, num_classes)

if __name__ == "__main__":
    main()
