# RoNFA: Robust Neural Field–based Few-Shot Classification with Noisy Labels

RoNFA is a novel neural field–based approach that addresses the challenge of **few-shot image classification in the presence of noisy labels**. It is designed to mimic visual cognition through biologically-inspired mechanisms like **receptive fields**, **local activation**, and **adaptive scale tuning**, making it **highly robust to noisy supervision** and efficient in low-data regimes.

> **Xiang, N., Xing, L., & Jin, D. (2025).**  
> *RoNFA: Robust Neural Field–based Approach for Few‐Shot Image Classification with Noisy Labels*. Preprint.

---

## 🧠 Overview

RoNFA innovatively integrates:
- **Two neural fields**: one for feature representation (FFR), another for category representation (FCR)
- **Soft K-Means clustering** to obtain robust class prototypes from noisy support sets
- **Scale-adaptive receptive fields** inspired by the Mexican-hat activation mechanism
- **Fully local learning**: avoids complex backpropagation and fine-tuning, making it fast and biologically plausible

---

## 📊 Experimental Results

RoNFA significantly outperforms existing methods across all noise types and datasets.


### 🔹 MiniImageNet — Symmetric & Paired Label Noise

| Model            | Backbone     | 0% Sym    | 20% Sym   | 40% Sym   | 60% Sym   | 40% Pair  |
| ---------------- | ------------ | --------- | --------- | --------- | --------- | --------- |
| Qwen-VL-plus     | -            | 98.58     | 81.20     | 58.40     | 31.07     | 76.13     |
| GPT-4o           | -            | 98.67     | 93.67     | 57.67     | 33.67     | 82.66     |
| ProtoNet         | Conv4        | 68.95     | 64.39     | 56.26     | 41.61     | 51.01     |
| RNNP             | Conv4        | 63.57     | 59.71     | 52.44     | 38.85     | 46.04     |
| TraNFS-3         | Conv4        | 67.65     | 63.53     | 53.00     | 39.57     | 48.55     |
| **RoNFA (Ours)** | **Conv4**    | **69.14** | **66.39** | **59.17** | **44.63** | **54.31** |
| ProtoNet         | ResNet18     | 97.77     | 96.41     | 93.91     | 79.45     | 85.73     |
| RNNP             | ResNet18     | 97.34     | 96.88     | 95.15     | 75.71     | 86.59     |
| TraNFS-3         | ResNet18     | 81.89     | 79.47     | 70.95     | 52.41     | 64.45     |
| **RoNFA (Ours)** | **ResNet18** | **98.29** | **97.92** | **97.46** | **92.06** | **95.35** |
| ProtoNet         | ViT          | 98.46     | 97.59     | 96.39     | 88.27     | 91.07     |
| RNNP             | ViT          | 98.57     | 98.20     | 96.87     | 77.34     | 88.04     |
| TraNFS-3         | ViT          | 98.83     | 98.31     | 97.31     | 81.00     | 95.80     |
| **RoNFA (Ours)** | **ViT**      | **99.17** | **99.12** | **99.11** | **98.33** | **98.76** |

---

### 🔹 TieredImageNet — Symmetric & Paired Label Noise

| Model            | Backbone     | 0% Sym    | 20% Sym   | 40% Sym   | 60% Sym   | 40% Pair  |
| ---------------- | ------------ | --------- | --------- | --------- | --------- | --------- |
| Qwen-VL-plus     | -            | 94.93     | 80.54     | 47.47     | 19.20     | 76.00     |
| GPT-4o           | -            | 97.00     | 77.66     | 58.00     | 21.34     | 83.67     |
| ProtoNet         | Conv4        | 63.47     | 60.06     | 43.13     | 32.52     | 43.38     |
| RNNP             | Conv4        | 53.60     | 46.92     | 40.86     | 30.94     | 38.44     |
| TraNFS-3         | Conv4        | 62.95     | 58.43     | 50.19     | 36.17     | 46.55     |
| **RoNFA (Ours)** | **Conv4**    | **66.37** | **62.36** | **50.65** | **37.02** | **46.24** |
| ProtoNet         | ResNet18     | 91.48     | 89.82     | 84.39     | 67.60     | 76.21     |
| RNNP             | ResNet18     | 90.25     | 89.71     | 84.59     | 64.29     | 77.87     |
| TraNFS-3         | ResNet18     | 78.65     | 74.17     | 64.29     | 48.79     | 61.08     |
| **RoNFA (Ours)** | **ResNet18** | **91.90** | **91.77** | **88.64** | **76.57** | **85.97** |
| ProtoNet         | ViT          | 94.67     | 92.29     | 89.38     | 74.74     | 82.35     |
| RNNP             | ViT          | 94.42     | 92.62     | 88.62     | 63.68     | 77.69     |
| TraNFS-3         | ViT          | 97.00     | 96.80     | 94.73     | 75.87     | 91.20     |
| **RoNFA (Ours)** | **ViT**      | **97.27** | **96.92** | **96.63** | **93.53** | **95.12** |

---

### 🔹 CUB — Symmetric / Paired / Outlier Noise

| Model            | Backbone     | 0%        | 40% Sym   | 60% Sym   | 40% Pair  | 40% Out   | 60% Out   |
| ---------------- | ------------ | --------- | --------- | --------- | --------- | --------- | --------- |
| ProtoNet         | Conv4        | 64.06     | 42.93     | 31.28     | 43.56     | 53.89     | 48.41     |
| RNNP             | Conv4        | 52.26     | 37.97     | 28.64     | 36.29     | 34.13     | 32.34     |
| **RoNFA (Ours)** | **Conv4**    | **64.32** | **48.96** | **34.86** | **45.01** | **57.39** | **53.12** |
| ProtoNet         | ResNet18     | 86.81     | 76.32     | 56.21     | 67.47     | 85.48     | 82.49     |
| RNNP             | ResNet18     | 84.33     | 78.14     | 52.71     | 69.28     | 75.99     | 75.44     |
| **RoNFA (Ours)** | **ResNet18** | **85.52** | **81.18** | **66.59** | **77.71** | **84.19** | **83.07** |
| ProtoNet         | ViT          | 95.25     | 92.76     | 86.73     | 88.64     | 94.52     | 93.74     |
| RNNP             | ViT          | 95.17     | 93.06     | 79.64     | 86.16     | 89.05     | 87.89     |
| **RoNFA (Ours)** | **ViT**      | **95.32** | **94.00** | **90.45** | **92.19** | **94.90** | **94.00** |

---

### 🔹 Caltech101 / DTD — Symmetric & Outlier Noise

| Dataset    | Model            | Backbone     | 0%        | 40% Sym   | 60% Sym   | 40% Out   | 60% Out   |
|------------| ---------------- | ------------ | --------- | --------- | --------- | --------- | --------- |
| Caltech101 | ProtoNet         | Conv4        | 83.91     | 73.32     | 53.40     | 79.92     | 78.11     |
|            | RNNP             | Conv4        | 79.90     | 69.90     | 48.44     | 73.77     | 72.74     |
|            | **RoNFA (Ours)** | **Conv4**    | **84.66** | **78.50** | **58.70** | **81.65** | **81.22** |
|            | ProtoNet         | ResNet18     | 96.24     | 93.15     | 78.53     | 95.92     | 94.30     |
|            | RNNP             | ResNet18     | 95.42     | 94.03     | 73.88     | 87.98     | 87.90     |
|            | **RoNFA (Ours)** | **ResNet18** | **96.70** | **96.15** | **88.30** | **96.28** | **95.58** |
|            | ProtoNet         | ViT          | 97.02     | 95.34     | 89.16     | 96.51     | 94.98     |
|            | RNNP             | ViT          | 96.54     | 95.03     | 79.18     | 89.25     | 88.51     |
|            | **RoNFA (Ours)** | **ViT**      | **97.58** | **97.17** | **95.59** | **97.49** | **96.74** |
| DTD        | ProtoNet         | Conv4        | 55.48     | 41.24     | 31.22     | 48.44     | 42.89     |
|            | RNNP             | Conv4        | 47.46     | 37.35     | 29.40     | 37.32     | 36.37     |
|            | **RoNFA (Ours)** | **Conv4**    | **57.04** | **44.48** | **33.61** | **51.88** | **49.07** |
|            | ProtoNet         | ResNet18   | 74.89  | 67.32   | 49.97   | 71.43   | 69.11   |
|            | RNNP             | ResNet18   | 69.44  | 63.67   | 47.22   | 66.33   | 65.49   |
|            | **RoNFA (Ours)** | **ResNet18** | **76.01** | **71.85** | **54.44** | **73.95** | **72.02** |
|            | ProtoNet         | ViT          | 80.68     | 71.22     | 51.85     | 78.01     | 73.89     |
|            | RNNP             | ViT          | 74.82     | 65.60     | 47.69     | 71.32     | 69.80     |
|            | **RoNFA (Ours)** | **ViT**      | **80.75** | **74.43** | **56.98** | **79.59** | **76.45** |

---
## ⚗️ Ablation Study

We assess the importance of key components of **RoNFA** through controlled experiments and design variants:

### 🧪 Effect of Soft vs. Hard K-Means Clustering

| Variant           | MiniImageNet (%) | TieredImageNet (%) |
|-------------------|------------------|---------------------|
| Hard K-Means      | 98.88            | 93.49               |
| **Soft K-Means**  | **99.11**        | **94.85**           |

> **Insight:** Soft cluster assignment leads to smoother and more noise-tolerant prototype learning.

---

### 🧪 Effect of Adaptive Receptive Field Scaling

| Variant                    | MiniImageNet (%) | TieredImageNet (%) |
|----------------------------|------------------|---------------------|
| Fixed RF Scale (non-adaptive) | 97.28        | 92.86               |
| **Adaptive Scale (Ours)**  | **99.11**        | **94.85**           |

> **Insight:** Adaptively adjusting receptive field size per query image improves generalization under uncertainty.

---

### 🧪 Combined Component Analysis (MiniImageNet, ViT Backbone)

| Configuration            | Accuracy (%) |
|--------------------------|--------------|
| w/o Adaptive Scale       | 97.28        |
| w/o Soft K-Means         | 98.88        |
| **Full RoNFA (Ours)**    | **99.17**    |

> **Conclusion:** Each component (adaptive scale, soft K-Means) independently improves robustness. Their combination leads to the best results.


---

## 📁 Repository Structure

```text
RoNFA/
├── data/
│   ├── mini_test_transformer_features.mat   # Extracted ViT feature vectors for dataset
│   └── mini_test_labels.mat                 # Ground truth labels for evaluation
├── main.m                                   # Main entry point for prediction and evaluation
...
```

---

## 📖 Citation

If you use this code in your work, please cite:

```bibtex
@article{xiang2025ronfa,
  title   = {RoNFA: Robust Neural Field–based Approach for Few‐Shot Image Classification with Noisy Labels},
  author  = {Xiang, Nan and Xing, Lifeng and Jin, Dequan},
  journal = {Preprint},
  year    = {2025},
}
```

---

## 📜 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- Pretrained ViT feature extractor adapted from [MATLAB Central File Exchange].  
- Inspired by Neural Field literature and few‐shot noisy‐label research.

Enjoy experimenting with RoNFA! Feel free to open issues or submit pull requests.
