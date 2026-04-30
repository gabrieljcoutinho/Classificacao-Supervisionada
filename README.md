# 📊 Checkpoint 03 — DSA
## Classificação Supervisionada: Dados Sintéticos e Dados Reais

Este projeto demonstra a aplicação de técnicas de **Machine Learning supervisionado** em dois cenários:

1. **Dataset sintético (não linear)** → usando redes neurais (MLP)
2. **Dataset real (fraude de cartão)** → usando múltiplos modelos clássicos

O objetivo é comparar desempenho, entender impacto de pré-processamento e analisar dificuldades reais dos dados.

---

# 🚀 Tecnologias utilizadas

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Imbalanced-learn (SMOTE)

---

# 📁 Estrutura do Projeto
parte1_dataset.png
├── parte1_loss_curve.png
├── parte1_decision_boundary.png
├── parte1_confusion_matrix.png
├── parte2_curvas.png
├── parte2_confusion_matrix.png
├── creditcard.csv
└── main.py


---

# 🧠 PARTE 1 — Classificação com Dados Sintéticos

## 🎯 Objetivo
Resolver um problema de classificação binária **não linearmente separável** usando uma rede neural (MLP).

---

## 📌 Dataset

- Gerado com `make_moons`
- 1000 amostras
- Ruído: 0.25
- 2 classes

Esse dataset simula um problema onde **uma linha reta não separa os dados**.

---

## ⚙️ Pipeline

1. Geração dos dados
2. Split treino/teste (80/20)
3. Normalização com `StandardScaler`
4. Teste de múltiplas arquiteturas de MLP
5. Avaliação de desempenho

---

## 🔬 Modelos testados

- Diferentes quantidades de camadas:
  - 1 camada
  - 2 camadas
  - 3 camadas
- Funções de ativação:
  - ReLU
  - Tanh
- Taxas de aprendizado
- Com e sem normalização

---

## 📈 Métricas avaliadas

- Acurácia (treino e teste)
- Classification Report
- Matriz de confusão

---

## 📊 Visualizações geradas

- Distribuição dos dados
- Curva de loss (treinamento)
- Fronteira de decisão
- Matriz de confusão

---

## 💡 Principais insights

- Normalização melhora significativamente o desempenho
- Redes simples já resolvem o problema
- Mais camadas ≠ melhor resultado
- Risco de overfitting ao aumentar complexidade

---

# 💳 PARTE 2 — Detecção de Fraude (Dataset Real)

## 🎯 Objetivo
Detectar transações fraudulentas em um dataset real altamente desbalanceado.

---

## 📌 Dataset

- Credit Card Fraud Detection (Kaggle)
- ~285 mil transações
- Apenas ~0.17% são fraudes

---

## ⚠️ Desafio principal

**Desbalanceamento extremo de classes**

> Um modelo pode ter 99.8% de acurácia e ainda ser inútil

---

## ⚙️ Pipeline

1. Carregamento dos dados
2. Separação de features e target
3. Normalização de `Amount` e `Time`
4. Split treino/teste
5. Aplicação de **SMOTE** para balanceamento
6. Treinamento de múltiplos modelos

---

## 🤖 Modelos utilizados

- Regressão Logística
- Árvore de Decisão
- Random Forest
- Gradient Boosting

---

## 📈 Métricas avaliadas

- ROC-AUC
- Average Precision (PR-AUC) ⭐ principal
- F1-score (classe fraude)

---

## 📊 Visualizações

- Curva ROC
- Curva Precision-Recall
- Matriz de confusão

---

## 🧠 Por que usar Average Precision?

Porque em datasets desbalanceados:

- Acurácia engana
- ROC pode ser otimista
- **Precision-Recall mostra o desempenho real na classe rara**

---

## 💡 Principais insights

- O problema é difícil devido ao desbalanceamento
- SMOTE melhora a detecção de fraudes
- Modelos ensemble (Random Forest / Boosting) tendem a performar melhor
- Métricas corretas são essenciais

---

# 🛠️ Como executar

```bash
pip install numpy pandas matplotlib scikit-learn imbalanced-learn
