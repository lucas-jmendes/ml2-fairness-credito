# Registro de decisões técnicas e de escopo

Cada entrada: data, decisão, justificativa, status (proposta pela IA / aprovada pelo grupo).

## 20/11/2026 — Escopo enxuto
Ver CLAUDE.md. Status: [A COMPLETAR: aprovação do grupo].

## 20/11/2026 — Núcleo do código e braço A0 (proposto pelo Claude Code, aguardando revisão)

1. **AGE fora das entradas, além de SEX.** A proposta (seção 5) diz que os atributos sensíveis, inclusive idade, ficam fora das variáveis de entrada. Com isso, o modelo usa 21 variáveis.
   - Ponto em aberto: `MARRIAGE` (estado civil) continua como entrada. A ECOA também protege o estado civil, então um revisor pode cobrar a remoção. A decisão é do grupo.
2. **Recodificação de códigos não documentados.** `EDUCATION` 0, 5 e 6 viram 4 ("outros"); `MARRIAGE` 0 vira 3 ("outros"). Essa é a prática comum com esta base. Para os modelos de árvore a mudança é irrelevante; para a regressão logística, evita categorias raras no one-hot.
3. **Limiar de decisão em A0.** O limiar é escolhido em cada trial como o que maximiza a acurácia balanceada na validação (busca exata na curva ROC). Ele não entra como hiperparâmetro do Optuna, para não gastar orçamento com ele. Em A1 e A2 o limiar entra no espaço de busca (proposta). Decidir até a implementação de A1 como parametrizá-lo; sugestão: taxa de predições positivas em vez de probabilidade bruta, porque as escalas de probabilidade variam entre modelos.
4. **Objetivo de A0: acurácia balanceada na validação, com o TPE.** Para objetivo único, o TPE é o amostrador usual. O NSGA-II fica para os braços multiobjetivo. Assim o A0 tem sua melhor chance (baseline forte).
5. **CatBoost com profundidade máxima 8 (padrão da biblioteca: 6).** Com profundidade 10 e 500 iterações, um único treino levou 38 s, contra menos de 3 s dos demais modelos. Isso inviabilizaria o A2 no orçamento previsto.
6. **Random Forest: até 200 árvores, subamostras de 20% a 30%.** Conforme a seção 5 da proposta.
7. **Sem early stopping nos modelos de boosting.** Ele exigiria uma quarta partição ou usaria a validação duas vezes. O número de iterações é um hiperparâmetro da busca.
8. **O melhor trial é retreinado no treino (não em treino+validação).** O limiar e, depois, o pós-processamento (A4/A5) são ajustados na validação. Retreinar em treino+validação quebraria essa separação.
9. **Escores de validação e teste salvos** em `results/scores/`, como insumo para A4/A5 e para os intervalos de bootstrap sem retreino.

## 20/11/2026 — Mudança de escopo: idade em Taiwan e viabilidade da HMDA (proposto pelo Claude Code, solicitado pelo Lucas; aguardando aprovação do grupo)

**Motivação.** No A0 (semente 0, primeira divisão), a violação de EO por sexo no teste ficou entre 0,02 e 0,04 em todos os modelos. Com isso, os níveis ε = 0,05 e 0,10 da H3 ficam sem conteúdo. Calculada nos mesmos escores, a violação por idade < 25 ficou entre 0,10 e 0,15.

**Cuidado metodológico.** Os dois atributos de Taiwan (sexo e idade) e a HMDA com raça já estavam na proposta aprovada. Nenhum atributo é trocado depois de ver o resultado: todos são reportados, inclusive o viés baixo por sexo, que é um achado em si.

10. **Idade < 25 como segundo atributo sensível em Taiwan**, só na avaliação (`AGE` já estava fora das entradas). A divisão passa a ser estratificada por rótulo × sexo × faixa etária, então os resultados do A0 da primeira divisão foram descartados e o A0 foi refeito.
11. **HMDA: Geórgia, 2024.** O estado tem poder estatístico (cerca de 76 mil solicitantes negros nos dados brutos). Escolhi 2024 e não 2025 porque os agregados de 2025 trouxeram uma soma anômala (cerca de US$ 1 trilhão em pedidos negados).
12. **HMDA: grupos raciais branco e negro** (`derived_race`). Pedidos "Joint", outras raças e raça não informada ficam de fora. Isso maximiza o poder estatístico do contraste mais estudado em crédito justo e vai para as limitações.
13. **HMDA: filtros de população** — primeira hipoteca, residência principal, sem fim comercial, sem crédito rotativo, sem hipoteca reversa. Rótulo: negado (3) = 1; contratado (1) ou aprovado e não aceito (2) = 0.
14. **HMDA: variáveis da proposta** — valor, renda, dívida/renda, LTV, tipo, finalidade, prazo. Excluí campos posteriores à decisão (vazamento) e geográficos (proxy direto de raça). A dívida/renda vem em faixas e virou ordinal pelo ponto médio.
15. **HMDA: subamostra de 50 mil pedidos** (a proposta previa cerca de 200 mil), estratificada por rótulo × raça, com semente fixa 2026, por custo computacional.
16. **Valores ausentes:** a regressão logística imputa a mediana e ganha indicadores de ausência; os modelos de árvore tratam ausentes nativamente.
17. **Proteção de retomada:** cada estudo do Optuna guarda a impressão digital da divisão, das variáveis e do espaço de busca.

**Risco identificado (HMDA).** O LTV está ausente em 11% dos pedidos, e entre eles a taxa de negação é 47,8%, contra 17,7% nos demais. A ausência provavelmente reflete o processo (pedidos negados antes da avaliação do imóvel), não o risco: é um vazamento parcial do rótulo. Proposta: análise de sensibilidade sem esses pedidos, ou sem o indicador de ausência.

## 20/11/2026 — Recomendações aceitas pelo Lucas (pendente: líder técnico)

18. **HMDA promovida a base principal; atributos otimizados em A1/A2:** raça na HMDA e idade < 25 em Taiwan; o sexo é reportado como secundário nas duas bases. CLAUDE.md atualizado.
19. **Sensibilidade do LTV:** `configs/hmda_sens_ltv.yaml` herda de `hmda.yaml` e só descarta os pedidos com LTV ausente antes da subamostra. Roda o A0 com as mesmas 5 sementes.
20. **"Joint" verificado empiricamente** (a documentação pública não traz a regra; ver `scripts/hmda_population_report.py` e `results/data/`). Raça: em 90,3% dos casos, um branco e outro de minoria. Sexo: homem + mulher, ou os dois sexos marcados.
21. **Bug de reprodutibilidade corrigido:** a API do CFPB recusa (403) User-Agents personalizados. O download pelo Python passou a usar o padrão. Verificado: o arquivo baixado pelo script tem o mesmo SHA-256 registrado.
22. **Impressão digital dos estudos** passou a incluir o hash do conteúdo dos dados. Assim, variantes com a mesma divisão e dados diferentes não se misturam. Os estudos da semente 0 foram refeitos, e conferimos se reproduzem os resultados anteriores.
23. **Repositório consolidado:** `docs/` e `CLAUDE.md` foram movidos para dentro de `ml2-fairness-credito/`. `docs/artigos/` fica fora do git (direitos autorais).
