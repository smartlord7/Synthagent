# Resposta aos comentários (draft ICDE 2027, 7 out. 2026)

Cada comentário do PDF anotado e o que mudou no manuscrito.

## Página 1: Resumo

| Comentário | Alteração |
|---|---|
| Confundo "small data" com classes minoritárias; small data = "centenas de linhas"? | Small data passa a ser definido em relação à tarefa, para o dataset inteiro (Definição 1, do nosso survey): índice de pequenez σ = n★/n_eff. O desequilíbrio entra apenas através de n★. A Tabela III dá n★ e σ para os seis datasets. |
| Uniformizar o termo (small tables / small data / small-data contexts) | "Small data" é o regime e "small dataset" um dataset concreto; os outros termos foram eliminados. |
| "Não é um romance" | Frases literárias removidas ("a raw completion is not a data product", "every row counts", "control plane", etc.). O título passa a ser descritivo. |
| Não se percebe o contexto, porquê LLMs e qual é o problema | O resumo segue contexto → problema → porquê LLMs (trazem conhecimento externo; um gerador ajustado só aos dados não traz) → abordagem → resultados. |
| Demasiados termos por metro quadrado | O resumo já não fala de JSD, C2ST, AUC, noise floor nem do número de ajustes. |
| Porquê medicina, porquê estes 6 datasets, porquê 10 classificadores | Justificado na Secção V (Datasets, Protocol, Classifiers). |
| O que significa "complete"? | Definido: percentagem de valores não vazios (99,7–100%). |
| Contribuição pouco clara; parece ML e não sistema; resultados fracos | O sistema é apresentado pelos requisitos que garante (R1 tamanho exato, R2 esquema validado, R3 custo medido), sem treino nos dados. Os resultados mostram robustez: com GPT-4.1, o maior ganho médio de F1 e uma perda máxima de 0,53 pp em qualquer dataset, contra perdas de até 8,2 (SMOTE) e 10,6 (CTGAN). O enquadramento pela Proposição 4.1 do survey explica porque não se devem esperar ganhos médios grandes. |

## Página 1: Introdução

| Comentário | Alteração |
|---|---|
| "consequential", "small-data problems" sem definição | Removido; a introdução começa pelo contexto e pela Definição 1. |
| Repete a frase do resumo; porquê "labeled"? | A introdução tem texto próprio. O cenário supervisionado é dito explicitamente e a Definição 1 explica o que significa "labelled". |
| Porquê o split? | Explicado na Secção V-B: a utilidade tem de ser medida em dados reais que o gerador não viu; repete-se dez vezes porque um único split de um dataset pequeno é instável. |
| "Atrativo" sem explicar o problema | O problema vem antes, num parágrafo próprio, seguido de um Problem statement formal (R1–R6). |
| Frases "filosóficas" (proposal not a dataset, control plane) | Removidas. |
| "Isto é a maneira como implementaste o sistema" | A descrição da implementação passou para a Secção IV; a introdução diz apenas o que a abordagem faz e porquê. |
| Parágrafo atacável sem justificação ("Why small data changes the evaluation") | Removido; a justificação ficou na limitação L4 e no protocolo. |

## Página 2

| Comentário | Alteração |
|---|---|
| As 4 contribuições não decorrem do texto | Três contribuições, cada uma ligada a um requisito e a uma secção. |
| O que é a control layer? | Definida no parágrafo "Our approach", com os seus quatro componentes, e usada sempre com o mesmo nome. |
| "Key findings" é só uma descrição dos resultados | Passou a "Main findings": três afirmações demonstradas, cada uma ligada a uma RQ. |
| "A secção XXX concludes" | Parágrafo de organização removido. |
| Frase de romance na abertura do Related Work | Removida. |
| Estrutura do estado da arte (áreas → abordagens para small data → limitações → o que endereçamos) | A Secção II segue exatamente essa estrutura: quatro famílias de SDG, limitações L1–L4 para small data, "o Synthagent endereça L3 e L4 e evita L1 e L2", e a Tabela I. |

## Página 3

| Comentário | Alteração |
|---|---|
| A formulação do problema tem de aparecer na introdução; primeiro definir small data e estratégias, depois LLMs | A ordem da introdução é: Definição 1 → estratégias (canais de proveniência do survey) → porquê LLMs → problema → Problem statement. A Secção III apenas formaliza. |
| "Porquê isto é importante/melhor/inovador?" (definição operacional) | Substituída pela definição formal do survey, que é quantitativa e relativa à tarefa. |
| "Resolves tudo com validação"; o multiobjetivo aparece do nada | O parágrafo dos três riscos foi removido. O multiobjetivo decorre agora diretamente de R1–R6: R1–R2 são restrições garantidas pelo sistema; R3–R6 são objetivos medidos, e explica-se porque entram em conflito. |

## Página 6

| Comentário | Alteração |
|---|---|
| As RQs têm de estar na introdução | Passaram para a introdução. |
| As RQs têm de ilustrar o objetivo e estar ligadas entre si | Um objetivo explícito e quatro RQs encadeadas: fiabilidade → qualidade → utilidade → comparação. Os resultados estão organizados por RQ, cada uma com uma resposta explícita. |

## Outras alterações

- A secção e a figura de deployment federado foram retiradas do corpo (não eram avaliadas e desviavam a narrativa); o tema fica num parágrafo da Discussão.
- Foi adicionada a Tabela IX (variação de F1 por dataset, com o pior caso) para tornar os resultados legíveis.
- Os números são os mesmos; nenhum resultado foi alterado.
