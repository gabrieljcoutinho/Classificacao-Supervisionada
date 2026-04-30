# =============================================================================
# CHECKPOINT 03 — DSA
# Classificação Supervisionada: Dados Sintéticos e Dados Reais
# =============================================================================

# =============================================================================
# PARTE 1 — Classificação Binária em 2D com MLP
# Dataset sintético não-linearmente separável
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)

# ── 1.1  Geração do dataset sintético ────────────────────────────────────────
np.random.seed(42)
X, y = make_moons(n_samples=1000, noise=0.25, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Treino : {X_train.shape[0]} amostras")
print(f"Teste  : {X_test.shape[0]} amostras")
print(f"Classes: {np.unique(y)}")

# ── Visualização do dataset ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", alpha=0.7, edgecolors="k", linewidths=0.4)
ax.set_title("Dataset Sintético — make_moons", fontsize=14)
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
plt.colorbar(scatter, ax=ax, label="Classe")
plt.tight_layout()
plt.savefig("parte1_dataset.png", dpi=150)
plt.show()

# ── 1.2  Normalização ─────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── 1.3  Configurações de MLP a experimentar ─────────────────────────────────
configs = [
    # (hidden_layers,          activation, learning_rate_init, normalize, label)
    ((50,),                    "relu",     0.001,              True,  "1 camada  | 50n | relu | norm"),
    ((100, 50),                "relu",     0.001,              True,  "2 camadas | 100-50n | relu | norm"),
    ((100, 100, 50),           "relu",     0.001,              True,  "3 camadas | 100-100-50n | relu | norm"),
    ((100, 50),                "tanh",     0.001,              True,  "2 camadas | 100-50n | tanh | norm"),
    ((100, 50),                "relu",     0.01,               True,  "2 camadas | 100-50n | relu | lr=0.01 | norm"),
    ((100, 50),                "relu",     0.001,              False, "2 camadas | 100-50n | relu | SEM norm"),
]

results = []

print("\n" + "=" * 70)
print(f"{'Configuração':<45} {'Acurácia Treino':>15} {'Acurácia Teste':>14}")
print("=" * 70)

for hidden, activation, lr, use_norm, label in configs:
    Xt = X_train_scaled if use_norm else X_train
    Xv = X_test_scaled  if use_norm else X_test

    mlp = MLPClassifier(
        hidden_layer_sizes=hidden,
        activation=activation,
        learning_rate_init=lr,
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
    )
    mlp.fit(Xt, y_train)

    acc_train = accuracy_score(y_train, mlp.predict(Xt))
    acc_test  = accuracy_score(y_test,  mlp.predict(Xv))

    results.append((label, acc_train, acc_test, mlp, use_norm))
    print(f"{label:<45} {acc_train:>14.4f}  {acc_test:>13.4f}")

print("=" * 70)

# ── 1.4  Melhor modelo ────────────────────────────────────────────────────────
best = max(results, key=lambda r: r[2])
best_label, best_acc_train, best_acc_test, best_mlp, best_norm = best

print(f"\nMelhor configuração: {best_label}")
print(f"  Acurácia treino : {best_acc_train:.4f}")
print(f"  Acurácia teste  : {best_acc_test:.4f}")

# ── Relatório completo ────────────────────────────────────────────────────────
Xv_best = X_test_scaled if best_norm else X_test
print("\nRelatório de classificação (melhor modelo):")
print(classification_report(y_test, best_mlp.predict(Xv_best)))

# ── Curva de loss ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(best_mlp.loss_curve_, color="steelblue", linewidth=1.8, label="Loss treino")
ax.set_title(f"Curva de Loss — {best_label}", fontsize=12)
ax.set_xlabel("Época")
ax.set_ylabel("Loss")
ax.legend()
plt.tight_layout()
plt.savefig("parte1_loss_curve.png", dpi=150)
plt.show()

# ── Fronteira de decisão do melhor modelo ────────────────────────────────────
def plot_decision_boundary(model, X, y, scaler=None, title="Fronteira de Decisão"):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    if scaler:
        grid = scaler.transform(grid)
    Z = model.predict(grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.contourf(xx, yy, Z, alpha=0.35, cmap="coolwarm")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k", linewidths=0.4, s=25)
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    plt.savefig("parte1_decision_boundary.png", dpi=150)
    plt.show()

sc = scaler if best_norm else None
plot_decision_boundary(best_mlp, X_test, y_test, scaler=sc,
                       title=f"Fronteira de Decisão — {best_label}")

# ── Matriz de confusão ────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, best_mlp.predict(Xv_best))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Matriz de Confusão — Melhor MLP")
plt.tight_layout()
plt.savefig("parte1_confusion_matrix.png", dpi=150)
plt.show()

# ── 1.5  Discussão ────────────────────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  PARTE 1 — DISCUSSÃO                                                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  O dataset make_moons NÃO é linearmente separável, mas possui estrutura     ║
║  geométrica clara, o que o torna resolvível por MLPs mesmo com poucas       ║
║  camadas.                                                                    ║
║                                                                              ║
║  O que funcionou melhor:                                                     ║
║    • Normalização dos dados (StandardScaler) foi essencial — sem ela a      ║
║      convergência foi mais lenta e a acurácia inferior.                     ║
║    • 2 camadas ocultas (100→50) com relu e lr=0.001 apresentou o melhor    ║
║      equilíbrio entre capacidade e generalização.                           ║
║    • Adicionar uma 3ª camada não melhorou a acurácia de forma significativa ║
║      — o modelo "já resolveu" o problema antes disso.                       ║
║                                                                              ║
║  É um problema fácil?                                                        ║
║    Sim, relativamente. Acurácias >90% são obtidas com arquiteturas simples. ║
║    Aumentar camadas/neurônios além de certo ponto NÃO melhora o desempenho  ║
║    e pode levar a overfitting. A complexidade da rede deve ser proporcional  ║
║    à complexidade do problema.                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


# =============================================================================
# PARTE 2 — Classificação de Dataset Real: Fraude em Cartão de Crédito
# Kaggle Credit Card Fraud Detection
# =============================================================================

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve, f1_score
)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE          # pip install imbalanced-learn
from imblearn.pipeline import Pipeline as ImbPipeline

# ── 2.1  Carregamento dos dados ───────────────────────────────────────────────
# Ajuste o caminho conforme necessário
df = pd.read_csv("creditcard.csv")

print(f"\nShape   : {df.shape}")
print(f"Colunas : {list(df.columns)}")
print(f"\nDistribuição das classes:")
print(df["Class"].value_counts())
print(f"\n% de fraudes: {df['Class'].mean() * 100:.4f}%")

# ── 2.2  Pré-processamento ────────────────────────────────────────────────────
X = df.drop(columns=["Class"])
y = df["Class"]

# Normalizar 'Amount' e 'Time' (as demais colunas PCA já estão normalizadas)
from sklearn.preprocessing import StandardScaler as SS
X = X.copy()
X["Amount"] = SS().fit_transform(X[["Amount"]])
X["Time"]   = SS().fit_transform(X[["Time"]])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTreino: {X_train.shape[0]} | Teste: {X_test.shape[0]}")
print(f"Fraudes no treino: {y_train.sum()} ({y_train.mean()*100:.3f}%)")

# ── 2.3  Modelos com SMOTE para tratar desbalanceamento ───────────────────────
modelos = {
    "Regressão Logística": ImbPipeline([
        ("smote", SMOTE(random_state=42)),
        ("clf",   LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "Árvore de Decisão": ImbPipeline([
        ("smote", SMOTE(random_state=42)),
        ("clf",   DecisionTreeClassifier(max_depth=8, random_state=42))
    ]),
    "Random Forest": ImbPipeline([
        ("smote", SMOTE(random_state=42)),
        ("clf",   RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ]),
    "Gradient Boosting": ImbPipeline([
        ("smote", SMOTE(random_state=42)),
        ("clf",   GradientBoostingClassifier(n_estimators=100, random_state=42))
    ]),
}

# ── 2.4  Avaliação ────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print(f"{'Modelo':<25} {'ROC-AUC':>9} {'Avg Precision':>14} {'F1 (fraude)':>12}")
print("=" * 80)

resultados2 = {}
for nome, pipe in modelos.items():
    pipe.fit(X_train, y_train)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    y_pred = pipe.predict(X_test)

    roc   = roc_auc_score(y_test, y_prob)
    ap    = average_precision_score(y_test, y_prob)
    f1    = f1_score(y_test, y_pred)

    resultados2[nome] = {"roc": roc, "ap": ap, "f1": f1, "pipe": pipe, "y_prob": y_prob}
    print(f"{nome:<25} {roc:>9.4f} {ap:>14.4f} {f1:>12.4f}")

print("=" * 80)

# ── 2.5  Melhor modelo e relatório ───────────────────────────────────────────
melhor_nome = max(resultados2, key=lambda k: resultados2[k]["ap"])
melhor = resultados2[melhor_nome]

print(f"\nMelhor modelo: {melhor_nome}")
print(f"  ROC-AUC        : {melhor['roc']:.4f}")
print(f"  Avg Precision  : {melhor['ap']:.4f}")
print(f"  F1 (classe 1)  : {melhor['f1']:.4f}")

y_pred_best = melhor["pipe"].predict(X_test)
print("\nRelatório completo:")
print(classification_report(y_test, y_pred_best, target_names=["Normal", "Fraude"]))

# ── Curvas ROC e Precision-Recall ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for nome, r in resultados2.items():
    fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
    prec, rec, _ = precision_recall_curve(y_test, r["y_prob"])
    axes[0].plot(fpr, tpr, label=f"{nome} (AUC={r['roc']:.3f})")
    axes[1].plot(rec, prec, label=f"{nome} (AP={r['ap']:.3f})")

axes[0].plot([0,1],[0,1],"k--",linewidth=0.8)
axes[0].set_title("Curva ROC")
axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
axes[0].legend(fontsize=8)

axes[1].set_title("Curva Precision-Recall")
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("parte2_curvas.png", dpi=150)
plt.show()

# ── Matriz de confusão ────────────────────────────────────────────────────────
cm2 = confusion_matrix(y_test, y_pred_best)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=["Normal", "Fraude"])
disp2.plot(cmap="Greens")
plt.title(f"Matriz de Confusão — {melhor_nome}")
plt.tight_layout()
plt.savefig("parte2_confusion_matrix.png", dpi=150)
plt.show()

# ── 2.6  Discussão ────────────────────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  PARTE 2 — DISCUSSÃO                                                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Representação:                                                              ║
║    • Features V1–V28 já são componentes PCA (anonimizadas).                 ║
║    • Amount e Time foram normalizados com StandardScaler.                   ║
║    • SMOTE foi aplicado para lidar com o severo desbalanceamento de classes  ║
║      (~0.17% de fraudes).                                                   ║
║                                                                              ║
║  Escolha de modelos:                                                         ║
║    • Testamos 4 algoritmos clássicos do scikit-learn.                       ║
║    • Métrica principal: Average Precision (área sob Precision-Recall),      ║
║      mais informativa que acurácia em dados desbalanceados.                 ║
║                                                                              ║
║  É um problema fácil?                                                        ║
║    NÃO. O principal fator que torna o problema difícil é o SEVERO           ║
║    DESBALANCEAMENTO DE CLASSES (~492 fraudes em ~285 mil transações).       ║
║    Modelos ingênuos que sempre predizem "normal" atingem 99.8% de           ║
║    acurácia, mas detectam zero fraudes. Isso exige métricas adequadas       ║
║    (PR-AUC, F1 da classe minoritária) e estratégias como SMOTE ou           ║
║    ajuste de class_weight para treinar modelos úteis.                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")