# Projeto final — Machine Learning 2 (CESAR School)

## Contexto
Artigo científico em português no template SBC, para submissão à REIC, com código aberto no GitHub.
Entrega final: **07/12/2026**.

Título: "Crédito Assertivo e Justo sem Atributos Sensíveis: Ensembles Otimizados em Múltiplos Objetivos para Mitigação de Viés Algorítmico".

Áreas da ementa: Ensemble Learning + Otimização de Modelos.

A proposta aprovada (`docs/proposta.pdf`) define as hipóteses (H1–H3), os braços experimentais, as métricas e o protocolo estatístico. As diretrizes da disciplina estão em `docs/diretrizes.pdf`.

## Escopo enxuto (decidido em 20/11/2026, por falta de tempo)
- Base principal: Default of Credit Card Clients (Taiwan, UCI). Atributo sensível: sexo.
- Braços obrigatórios: A0, A1, A2, A4, A5, A6. 5 sementes, 100 avaliações por estudo, NSGA-II.
- Condicionais (decidir em 27/11): A3, ablação com stacking, comparação de amostradores.
- Removido: reprodução no ACSIncome (vai para as limitações do artigo).

## Atualização de escopo (20/11/2026; aceita pelo Lucas, pendente de aprovação do líder técnico)
Ver `docs/decisoes.md`, itens 10–23.
- HMDA (Geórgia, 2024, brancos × negros, 50 mil pedidos) promovida a base principal. Atributo otimizado: raça; sexo secundário. Configuração em `configs/hmda.yaml`.
- Taiwan: idade < 25 passa a ser o atributo otimizado em A1/A2; sexo é reportado como secundário.
- Sensibilidade obrigatória da HMDA sem os pedidos com LTV ausente (`configs/hmda_sens_ltv.yaml`).

Qualquer mudança adicional de escopo deve ser explícita, justificada e registrada em `docs/decisoes.md`.

## Forma de trabalho
- O Claude escreve todo o código; o Lucas revisa e aprova as alterações.
- Toda entrega vem com um resumo curto: (1) o que foi feito, (2) as decisões técnicas e por quê, (3) o que o Lucas deve conferir na revisão, (4) como rodar. O Lucas precisa conseguir defender cada decisão nas orientações.
- Antes de rodar algo que demore mais de alguns minutos, avisar a estimativa de tempo e o comando, para o Lucas executar no terminal se preferir.
- Atuar como coorientador técnico: criticar decisões, apontar riscos metodológicos, não apenas concordar.
- Nunca inventar resultados, números, tabelas ou referências. Todo número do artigo vem de saídas reais em `results/`.
- Em textos do artigo, marcar lacunas como [A COMPLETAR: ...].

## Padrões de código
- Python; scikit-learn, XGBoost, LightGBM, CatBoost, Optuna, Fairlearn, error-parity. Dependências travadas em `requirements.txt`.
- Código modular em `src/`; scripts de execução em `scripts/`; notebooks só para análise.
- Sementes fixas, configurações em YAML (`configs/`), resultados brutos em CSV (`results/`).
- Estudos do Optuna em SQLite, retomáveis após interrupção.
- O atributo sensível nunca entra como variável de entrada em A0, A1, A2 e A6 — só na validação/avaliação.
- Dados brutos em `data/raw/` (fora do git); script de download em `scripts/`.
- O código deve rodar do zero num ambiente limpo (requisito de reprodutibilidade da disciplina).

## Registro de uso de IA (OBRIGATÓRIO — omissão zera o projeto)
Ao final de cada sessão de trabalho, acrescentar uma linha em `docs/registro_ia.md`:
Data | Ferramenta | Prompt (literal ou resumido fielmente) | O que a IA produziu | Decisão do grupo e justificativa.
O anexo deve declarar de forma explícita que o código foi integralmente gerado por IA (Claude Code), com revisão, execução e validação humanas.
