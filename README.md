# Small Area Estimation (SAE): Fay-Herriot Modeling

A statistical modeling pipeline implementing Small Area Estimation (SAE) techniques to stabilize direct survey estimates across regional domains with low sample representativity. 

##  Project Context
When working with national socio-demographic surveys, disaggregating indicators (e.g., labor informality, poverty) at lower geographic levels often yields unacceptably high Standard Errors (SE) or Coefficients of Variation (CV). This project resolves sample size limitations by borrowing strength from auxiliary administrative data through area-level models.

##  Mathematical Framework
The implementation relies on the **Fay-Herriot model**, linking direct survey estimates $\hat{\theta}_i$ to domain-specific auxiliary variables $\mathbf{x}_i$:

1. **Sampling Model:** $\hat{\theta}_i = \theta_i + e_i \quad \text{where} \quad e_i \sim N(0, \psi_i)$
2. **Linking Model:** $\theta_i = \mathbf{x}_i^T \boldsymbol{\beta} + v_i \quad \text{where} \quad v_i \sim N(0, \sigma_v^2)$

This yields the Empirical Best Linear Unbiased Predictor (EBLUP), which is a weighted average of the direct estimate and the synthetic regression estimate, effectively reducing estimator variance.

##  Stack Tecnológico
- **Language:** Python 3.x
- **Libraries:** `scipy.stats`, `statsmodels`, `numpy`, `pandas`
- **Methodologies:** Generalized Linear Models (GLM), Variance estimation, Restricted Maximum Likelihood (REML).

##  Impact & Results
- Successfully stabilized variance and reduced CVs below the 15% institutional threshold for vulnerable population domains.
- Provided statistically rigorous socio-economic indicators for localized decision-making.

## Reproducibility

```bash
git clone [https://github.com/TU_USUARIO/sae-statistical-modeling.git](https://github.com/TU_USUARIO/sae-statistical-modeling.git)
cd sae-statistical-modeling
pip install -r requirements.txt
python src/run_model.py
