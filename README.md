# Churn MLflow Pipeline

Pipeline de experimentos rastreável para previsão de churn de clientes. Utiliza MLflow para experiment tracking, comparação de modelos e versionamento via Model Registry.

---

## Visão Geral

Este projeto é a evolução do [churn-model-api](https://github.com/caiobnd/churn-model-api). O foco aqui não é apenas treinar um modelo — é **rastrear, comparar e versionar experimentos** de forma reprodutível.

Problema: sem rastreamento, cada retreino sobrescreve o anterior. Você perde o histórico de quais parâmetros geraram qual resultado.

Solução: MLflow registra automaticamente parâmetros, métricas e artefatos de cada Run — tudo comparável em uma interface visual.

---

## Experimentos Realizados

9 Runs no total, comparando 3 algoritmos com diferentes hiperparâmetros:

| Run | Algoritmo | Parâmetros principais | Recall | F1 |
|---|---|---|---|---|
| **lr_v1** ⭐ | Logistic Regression | class_weight=balanced, max_iter=2000 | **0.794** | **0.612** |
| lr_v2 | Logistic Regression | max_iter=3000 | ~0.79 | ~0.61 |
| lr_v3 | Logistic Regression | solver=saga, max_iter=2000 | ~0.79 | ~0.61 |
| xgb_v1 | XGBoost | n_estimators=100, lr=0.1, spw=2.77 | 0.75 | 0.60 |
| xgb_v2 | XGBoost | n_estimators=200, lr=0.05, spw=2.77 | 0.75 | 0.60 |
| xgb_v3 | XGBoost | n_estimators=300, lr=0.05, spw=2.77 | 0.75 | 0.60 |
| xgb_v4 | XGBoost | n_estimators=100, lr=0.1, spw=3.5 | 0.75 | 0.60 |
| rf_v1 | Random Forest | n_estimators=100, class_weight=balanced | 0.45 | 0.53 |
| rf_v2 | Random Forest | n_estimators=200, max_depth=10 | 0.45 | 0.53 |

**Modelo selecionado: `lr_v1`** — melhor Recall na classe minoritária (churn). Em problemas de churn, o custo de perder um cliente é maior que o de um falso positivo.

---

## Estrutura do Projeto

```
churn-mlflow-pipeline/
├── data/
│   └── .gitkeep
├── model/
│   └── .gitkeep
├── mlruns/
│   └── .gitkeep
├── cleaning.py
├── constants.py
├── model.py
├── train.py
├── requirements.txt
└── README.md
```

---

## Como Executar

### 1. Clone o repositório

```bash
git clone https://github.com/caiobnd/churn-mlflow-pipeline.git
cd churn-mlflow-pipeline
```

### 2. Configure o ambiente

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

### 3. Baixe o dataset

Baixe o [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) e coloque o CSV em `data/`.

### 4. Suba o servidor MLflow

```bash
mlflow ui
```

Acesse `http://localhost:5000` para visualizar os experimentos.

### 5. Execute o pipeline

```bash
python train.py
```

As Runs aparecerão automaticamente na UI do MLflow.

---

## Tecnologias Utilizadas

- **Python 3.12**
- **MLflow** — experiment tracking e Model Registry
- **scikit-learn** — Logistic Regression e Random Forest
- **XGBoost** — gradient boosting
- **pandas** — manipulação de dados
- **joblib** — serialização de modelos

---

## Próximos Passos

- [ ] Adicionar `StandardScaler` para features numéricas e avaliar impacto na Logistic Regression
- [ ] Implementar `GridSearchCV` para tuning automatizado de hiperparâmetros
- [ ] Adicionar detecção de drift com Evidently AI
- [ ] Integrar CI/CD com GitHub Actions