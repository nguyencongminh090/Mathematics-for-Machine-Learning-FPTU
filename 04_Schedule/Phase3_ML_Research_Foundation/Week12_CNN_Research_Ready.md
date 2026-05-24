# Week 12 — CNN Basics & Research Ready
**Dates:** August 10 – August 16, 2026
**Phase:** 3 — ML Research Foundation (FINAL WEEK)
**Books:** Deep Learning Cơ Bản VI Ch4–8 + d2l Ch7
**Tools:** Python, PyTorch

---

## Goal This Week
Learn CNN enough to read and implement research papers that use it.
End the 12 weeks by building a complete, working image classifier.
You are now research-ready.

---

## Key Concepts to Master
- [ ] Convolution operation: what it does to an image
- [ ] Filters/kernels: edge detection, feature extraction
- [ ] Pooling layers: MaxPool, AvgPool
- [ ] Fully connected layer after convolution
- [ ] Classic architecture: LeNet-5
- [ ] Training a CNN on a real dataset (MNIST or CIFAR-10)
- [ ] Reading a research paper: identify dataset, architecture, metrics, results

---

## Daily Breakdown (1–2 hrs/day)

| Day | Date | Task | Code Goal |
|-----|------|------|-----------|
| Mon | Aug 10 | Read DL Cơ Bản Ch4 (CNN introduction). Understand convolution visually. | Reading |
| Tue | Aug 11 | Implement 2D convolution from scratch with NumPy. Apply to a simple image. | Manual conv |
| Wed | Aug 12 | Build LeNet-5 in PyTorch using `nn.Conv2d`, `nn.MaxPool2d`. | PyTorch CNN |
| Thu | Aug 13 | Train LeNet-5 on MNIST. Reach >98% test accuracy. | Full training |
| Fri | Aug 14 | Read one real research paper related to your research topic. Identify: dataset, model, metrics. | Paper reading |
| Sat | Aug 15 | Adapt the CNN for your specific research problem (if image-related) OR write a summary of the full 12-week journey. | Applied work |
| Sun | Aug 16 | FINAL DAY: Run all notebooks from Weeks 5–12. Everything working? You're research-ready. | Final review |

---

## Starter Code — LeNet-5 in PyTorch
```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# LeNet-5
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# Load MNIST
transform = transforms.ToTensor()
train_set = torchvision.datasets.MNIST('.', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

model = LeNet5()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    total_loss = 0
    for X, y in train_loader:
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
```

---

## 12-Week Completion Checklist
You are research-ready when you can:
- [ ] Derive gradient descent from first principles
- [ ] Implement Linear Regression, Logistic Regression, PCA from scratch
- [ ] Explain Bayes' theorem with a real example
- [ ] Implement and train a neural network from scratch (NumPy)
- [ ] Build and train a CNN in PyTorch
- [ ] Read an ML research paper and understand: data, model, training, evaluation
- [ ] Apply any of the above to your specific research topic

---

## What Comes Next (After Week 12)
1. Read 2–3 papers in your specific research area
2. Reproduce a paper's results using your new skills
3. Start your own research project
4. Refer to ML Cơ Bản (Vũ Hữu Tiệp) for deeper algorithms (SVM, Clustering, etc.)
5. Refer to d2l for advanced DL architectures (RNN, Transformer, Attention)

---

## Practical Tip
Research is not about knowing everything — it's about being able to learn what you need, when you need it. The 12 weeks gave you the foundation. Now you have the tools to go further on your own.

---

## Progress Tracker
- [ ] Convolution operation understood
- [ ] Convolution from scratch (NumPy)
- [ ] LeNet-5 built in PyTorch
- [ ] Trained on MNIST (>98% accuracy)
- [ ] Research paper read and analyzed
- [ ] All 12-week notebooks verified working
- [ ] RESEARCH READY ✓
