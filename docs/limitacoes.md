# Limitações: rascunho para a seção do artigo

Rascunho de 20/11/2026, gerado pelo Claude Code, a revisar pelo grupo. Os números citados vêm de `results/`; o arquivo de origem aparece entre parênteses. As lacunas estão marcadas como [A COMPLETAR].

## 1. Recorte racial da HMDA (só brancos e negros)

Usamos o campo `derived_race` da base pública e mantivemos só os pedidos classificados como "White" e "Black or African American". Na Geórgia, em 2024, esses dois grupos somam **71,3%** dos pedidos com decisão (`results/data/hmda_racas.csv`). O restante ficou de fora:

| Raça derivada | Participação | Taxa de negação |
|---|---|---|
| Raça não informada | 19,9% | 30,7% |
| Asiática | 6,4% | 19,1% |
| Joint | 1,4% | 21,4% |
| Indígena, 2+ raças de minoria, havaiana, texto livre | 0,9% | 38%–73% |

- **"Raça não informada" é a exclusão mais séria.** É um em cada cinco pedidos, e a taxa de negação desse grupo (30,7%) fica entre a de brancos (21,8%) e a de negros (36,3%). A ausência da raça não é aleatória. Se esse grupo concentra solicitantes de minorias, a disparidade real pode estar subestimada.
- **O que é "Joint" (raça).** A documentação pública só diz que o campo é "derivado dos campos do solicitante e do co-solicitante" e não dá a regra. Verificamos empiricamente: em 90,3% dos pedidos "Joint", um dos dois é branco e o outro de uma raça de minoria, pelo primeiro campo de raça (`results/data/hmda_joint_raca.csv`). São casais ou co-solicitantes inter-raciais. Os casos restantes envolvem pessoas que declararam mais de uma raça.
- **Justificativa do recorte:** o contraste branco × negro é o mais estudado em crédito justo e o que tem poder estatístico (76 mil pedidos de solicitantes negros). Grupos pequenos, como indígenas, teriam intervalos de confiança largos demais para 50 mil pedidos.

## 2. Sexo na HMDA: o que "Joint" significa

No `derived_sex`, "Joint" corresponde a um solicitante e um co-solicitante de sexos diferentes (homem + mulher), ou a quem marcou os dois sexos (`results/data/hmda_joint_sexo.csv`). Duas pessoas do mesmo sexo herdam esse sexo; por exemplo, homem + homem = "Male". Na nossa amostra, "Joint" é cerca de 31% dos pedidos, com taxa de negação mais baixa (15,8%). Esses pedidos ficam **fora da métrica de sexo**, então a métrica compara pedidos individuais ou de pessoas do mesmo sexo. No teste, só 67,9% a 68,8% dos pedidos têm sexo definido, conforme a semente (`results/a0/hmda/split_seed*.csv`).

## 3. O rótulo da HMDA é a decisão do banco, não a inadimplência

O modelo aprende a reproduzir decisões históricas, que podem carregar a mesma discriminação que queremos medir. A violação de Equalized Odds é calculada em relação a esse rótulo. Um modelo "justo por EO" iguala as taxas de erro em relação às decisões passadas, não em relação ao risco real. Em Taiwan o rótulo é o desfecho observado (inadimplência), o que torna as duas bases complementares.

## 4. Variáveis de risco não observadas na HMDA

A base pública não tem escore de crédito nem histórico de pagamentos (só o *tipo* de modelo de escore). Parte da diferença de negação entre grupos pode vir de risco legítimo não observado; parte pode ser discriminação. Não conseguimos separar as duas. O estudo mede quanto um modelo sem o atributo sensível reproduz a disparidade, não a causa dela.

## 5. Ausência informativa do LTV (possível vazamento)

O LTV está ausente em cerca de 11% dos pedidos, e entre eles a taxa de negação é 47,8%, contra 17,7% nos demais. A explicação provável é o processo: pedidos negados cedo não chegam à avaliação do imóvel. A ausência, portanto, informa parte da decisão. Análise de sensibilidade (`configs/hmda_sens_ltv.yaml`, sem esses pedidos; 5 sementes; `results/a0/hmda_sens_ltv/aggregate_wide.csv` contra `results/a0/hmda/aggregate_wide.csv`): nos modelos de árvore, a violação de EO por raça **sobe** de 0,052–0,060 para 0,076–0,086. A ROC-AUC cai de 0,863–0,865 para 0,845–0,847. A ausência do LTV dava sinal preditivo, mas a disparidade racial não depende dela. Fica até maior sem esses pedidos.

## 6. Escolhas de pré-processamento

- **Dívida/renda (HMDA):** é publicada em faixas (<20%, 20–30%, 30–36%, 50–60%, >60%) e com valores exatos entre 36 e 49. Convertemos as faixas para o ponto médio aproximado, uma escala ordinal, e não a razão real.
- **Subamostra da HMDA:** 50 mil dos 146.820 pedidos elegíveis, estratificada por rótulo × raça (a proposta previa cerca de 200 mil), por custo computacional.
- **Representatividade da HMDA:** um único estado e ano. Os resultados não se generalizam diretamente para outros estados, anos ou mercados.
- **Códigos de Taiwan:** códigos não documentados (`EDUCATION` 0, 5, 6; `MARRIAGE` 0) foram agrupados em "outros".

## 7. Taiwan: viés por sexo pequeno e grupo jovem pequeno

- **Sexo:** sem mitigação, a violação de EO por sexo já é baixa e muda bastante só com a troca da divisão: média de 0,035 a 0,039 entre modelos, com desvio-padrão de 0,016 a 0,021 entre sementes (`results/a0/taiwan/aggregate_wide.csv`). Pouca mitigação é possível ou necessária. Reportamos esse resultado como achado, não como falha.
- **Idade < 25:** cerca de 9% dos clientes (537 no teste). As métricas desse grupo variam muito: EO médio de 0,075 a 0,112, com desvio-padrão de 0,031 a 0,048 entre sementes (`results/a0/taiwan/aggregate_wide.csv`). Por isso usamos 5 sementes e intervalos de bootstrap.
- **Estado civil (`MARRIAGE`):** continua como entrada. A ECOA também protege o estado civil. [A COMPLETAR: decisão do grupo.]
- **Contexto da base:** dados de 2005, de um único banco de Taiwan.

## 8. Grupos binários e interseccionalidade

A proposta previa uma análise interseccional raça × sexo. Com o recorte binário de raça e "Joint" fora da métrica de sexo, os grupos interseccionais ficam pequenos. [A COMPLETAR: decidir se a análise entra no artigo ou vira trabalho futuro.]

## 9. Critério de justiça e incompatibilidades

Otimizamos Equalized Odds, seguindo Kozodoi et al. (2022). Pelos resultados de impossibilidade (Kleinberg et al. 2017; Chouldechova 2017), quando as taxas-base diferem, EO e paridade demográfica não podem ser satisfeitos juntos. Na HMDA, a diferença de taxa-base é grande (cerca de 30% contra 16% de negação). Então reduzir o EO não iguala as taxas de aprovação. No A0, a diferença de paridade demográfica por raça fica entre 0,121 e 0,129, contra EO de 0,052 a 0,072 (`results/a0/hmda/aggregate_wide.csv`).

## 10. Validação do protocolo e orçamento

- **Sem o ACSIncome:** a reprodução de Cruz e Hardt (2024) nessa base foi removida por tempo. Não validamos o pipeline de pós-processamento contra os valores publicados por eles.
- **Orçamento:** 100 avaliações por estudo e 5 sementes. CatBoost com profundidade ≤ 8 e Random Forest com até 200 árvores sobre 20–30% dos dados, por custo. Orçamentos maiores podem mudar a ordem entre os modelos.

## 11. Contexto legal

A motivação legal (ECOA) é norte-americana; Taiwan e Brasil têm marcos diferentes. [A COMPLETAR: um parágrafo sobre LGPD (dados sensíveis, art. 5º, II) e decisões automatizadas (art. 20), se o grupo quiser aproximar do contexto brasileiro; conferir na fonte antes de citar.]
