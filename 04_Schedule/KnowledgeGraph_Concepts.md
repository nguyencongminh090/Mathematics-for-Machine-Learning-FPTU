# Knowledge Graph — Mathematical Concept Dependencies

> This graph shows how each concept builds on previous ones.
> Read it top-to-bottom: you must understand a node before the ones pointing away from it.
```mermaid
%%{init: {'theme': 'default', 'themeVariables': {'fontSize': '14px'}, 'flowchart': {'useMaxWidth': false}}}%%
graph TD
    %% ── FOUNDATIONS ──
    LA["📐 Linear Algebra<br/>(vectors, matrices)"]
    CALC["∂ Calculus<br/>(derivatives)"]
    PROB["🎲 Probability<br/>(distributions)"]

    %% ── WEEK 1: Linear Algebra Deep Dive ──
    subgraph W1["Week 1 — Linear Algebra"]
        NORM["Norms &<br/>Inner Products"]
        ORTH["Orthogonality &<br/>Projections"]
        EIGEN["Eigenvalues &<br/>Eigenvectors"]
        SVD["SVD<br/>(Singular Value<br/>Decomposition)"]
    end

    %% ── WEEK 2: Calculus & Optimization ──
    subgraph W2["Week 2 — Calculus & Optimization"]
        GRAD["Gradient &<br/>Partial Derivatives"]
        JACOBIAN["Jacobian<br/>(vector functions)"]
        CHAIN["Chain Rule"]
        BACKPROP["Backpropagation"]
        GD["Gradient Descent"]
    end

    %% ── WEEK 3: Probability & Statistics ──
    subgraph W3["Week 3 — Probability & Statistics"]
        BAYES["Bayes' Theorem"]
        GAUSS["Gaussian<br/>Distribution"]
        MLE["MLE<br/>(Max Likelihood<br/>Estimation)"]
        COV["Covariance<br/>Matrix"]
    end

    %% ── WEEK 4: Applications ──
    subgraph W4["Week 4 — Linear Regression & PCA"]
        LINREG["Linear<br/>Regression"]
        PCA["PCA<br/>(Dimensionality<br/>Reduction)"]
    end

    %% ── PHASE 2 ──
    subgraph P2["Phase 2 — Math to Code"]
        NUMPY["NumPy<br/>Implementation"]
        SCRATCH["From-Scratch<br/>Coding"]
    end

    %% ── PHASE 3 ──
    subgraph P3["Phase 3 — ML Research Foundation"]
        LOGREG["Logistic<br/>Regression"]
        SVM["SVM"]
        EVAL["Model<br/>Evaluation"]
        REG["Regularization<br/>L1 / L2"]
        NN["Neural<br/>Networks"]
        CNN["CNN"]
    end

    RESEARCH["🎓 Research<br/>Ready"]

    %% ── EDGES: Week 1 ──
    LA --> NORM
    LA --> EIGEN
    NORM --> ORTH
    EIGEN --> SVD

    %% ── EDGES: Week 2 ──
    CALC --> GRAD
    GRAD --> JACOBIAN
    GRAD --> GD
    JACOBIAN --> CHAIN
    CHAIN --> BACKPROP

    %% ── EDGES: Week 3 ──
    PROB --> BAYES
    PROB --> GAUSS
    GAUSS --> MLE
    LA --> COV
    PROB --> COV

    %% ── EDGES: Week 4 ──
    MLE --> LINREG
    GD --> LINREG
    ORTH --> LINREG
    ORTH --> PCA
    EIGEN --> PCA
    SVD --> PCA
    COV --> PCA

    %% ── EDGES: Phase 2 ──
    SVD --> NUMPY
    EIGEN --> NUMPY
    GD --> SCRATCH
    LINREG --> SCRATCH
    PCA --> SCRATCH
    BAYES --> SCRATCH

    %% ── EDGES: Phase 3 ──
    LINREG --> LOGREG
    GD --> LOGREG
    GAUSS --> SVM
    ORTH --> SVM
    LOGREG --> EVAL
    LINREG --> EVAL
    EVAL --> REG
    BACKPROP --> NN
    GD --> NN
    LOGREG --> NN
    NN --> CNN

    %% ── EDGES: Research ──
    LOGREG --> RESEARCH
    SVM --> RESEARCH
    EVAL --> RESEARCH
    REG --> RESEARCH
    NN --> RESEARCH
    CNN --> RESEARCH

    %% ── STYLES ──
    style LA fill:#4A90D9,color:#fff,stroke:#2c6fad
    style CALC fill:#4A90D9,color:#fff,stroke:#2c6fad
    style PROB fill:#4A90D9,color:#fff,stroke:#2c6fad

    style SVD fill:#7B68EE,color:#fff,stroke:#5a4fcf
    style BACKPROP fill:#7B68EE,color:#fff,stroke:#5a4fcf
    style BAYES fill:#7B68EE,color:#fff,stroke:#5a4fcf
    style PCA fill:#7B68EE,color:#fff,stroke:#5a4fcf
    style LINREG fill:#7B68EE,color:#fff,stroke:#5a4fcf

    style NUMPY fill:#E8A838,color:#fff,stroke:#c4841a
    style SCRATCH fill:#E8A838,color:#fff,stroke:#c4841a

    style NN fill:#E85D5D,color:#fff,stroke:#c43a3a
    style CNN fill:#E85D5D,color:#fff,stroke:#c43a3a
    style LOGREG fill:#E85D5D,color:#fff,stroke:#c43a3a
    style SVM fill:#E85D5D,color:#fff,stroke:#c43a3a

    style RESEARCH fill:#27AE60,color:#fff,stroke:#1a8a47
```

---

## Color Legend

| Color | Meaning |
|-------|---------|
| 🔵 Blue | Foundation concepts (must know first) |
| 🟣 Purple | Core ML math (Week 1–4) |
| 🟡 Orange | Phase 2 — coding skills |
| 🔴 Red | Phase 3 — ML models |
| 🟢 Green | Final goal |
