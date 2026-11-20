# ml2-fairness-credito

Comitês heterogêneos otimizados em múltiplos objetivos para crédito justo sem atributos sensíveis. Projeto final de Machine Learning 2 (CESAR School).

Artigo: *Crédito Assertivo e Justo sem Atributos Sensíveis: Ensembles Otimizados em Múltiplos Objetivos para Mitigação de Viés Algorítmico*.

## Ambiente

Python 3.12.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt        # versões diretas travadas + o pacote do projeto (src/faircredit)
pytest                                 # métricas conferidas contra o Fairlearn; divisão 60/20/20
```

`requirements-lock.txt` traz o ambiente completo (incluindo dependências transitivas).

## Dados

```bash
python scripts/download_taiwan.py      # UCI id 350 → data/raw/taiwan/ (SHA-256 conferido)
python scripts/download_hmda.py        # API do CFPB → data/raw/hmda/ e data/processed/
```

- **Taiwan** — Default of Credit Card Clients (Yeh e Lien, 2009): 30.000 clientes; rótulo = inadimplência no mês seguinte. Atributos sensíveis: sexo (principal) e idade < 25 anos (secundário). `SEX` e `AGE` ficam fora das variáveis de entrada e são usados apenas na avaliação.
- **HMDA** — Geórgia, 2024, solicitantes brancos e negros; primeira hipoteca para residência principal; rótulo = pedido negado. Subamostra estratificada de 50.000 pedidos. Atributos sensíveis: raça (principal) e sexo (secundário). Configuração em `configs/hmda.yaml`; as contagens de cada filtro ficam em `results/data/hmda_preparo.csv`.

## Experimentos

| Braço | Descrição | Script |
|---|---|---|
| A0 | Cinco modelos (regressão logística, Random Forest, XGBoost, LightGBM, CatBoost) ajustados só por desempenho | `scripts/run_a0.py` |

```bash
python scripts/run_a0.py --seeds 0           # uma semente
python scripts/run_a0.py                     # todas as sementes do YAML
python scripts/run_a0.py --config configs/hmda.yaml --seeds 0
python scripts/run_a0.py --config configs/hmda_sens_ltv.yaml   # sensibilidade: sem LTV ausente
python scripts/summarize_a0.py --dataset hmda                  # média ± dp entre sementes
python scripts/hmda_population_report.py                       # população excluída pelo recorte racial
```

Os estudos do Optuna ficam em SQLite (`results/optuna/`) e são retomados se a execução for interrompida. Cada estudo guarda uma impressão digital da divisão, das variáveis e do espaço de busca; se a configuração mudar, o estudo antigo não é retomado (apague o `.db` para recomeçar).

## Estrutura

```
configs/          bases, protocolo e espaços de busca (YAML); variantes herdam via base_config
docs/             proposta, decisões (decisoes.md), limitações, registro de uso de IA
src/faircredit/   código (dados, divisão, métricas, modelos, otimização, braços)
scripts/          download dos dados e execução dos braços
tests/            testes das métricas e da divisão
results/          resultados brutos (CSV) e estudos do Optuna (SQLite)
notebooks/        apenas análise
```

## Protocolo

- Divisão 60/20/20 estratificada pelo cruzamento rótulo × atributos sensíveis (Taiwan: sexo e faixa etária; HMDA: raça); a semente define a divisão, os modelos e o amostrador.
- Otimização na validação; avaliação final única no teste.
- Métricas: acurácia balanceada, ROC-AUC, violação de Equalized Odds (maior diferença entre grupos na TPR ou na FPR) e diferença de paridade demográfica.

## Uso de IA

O código deste repositório foi gerado integralmente com o Claude Code (Anthropic). A revisão, a execução e a validação foram feitas pela equipe. O registro completo está em `docs/registro_ia.md`.
