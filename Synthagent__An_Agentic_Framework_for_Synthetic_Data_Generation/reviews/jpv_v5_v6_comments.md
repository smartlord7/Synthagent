# Comentários do JPV (João P. Vilela) à v5 (até à Secção 3) e à v6 (Secções 3–4)

Ficheiros: `20261102-synthagent_revised_draft_v5 - jpv up to section 3.pdf` (27 comentários do JPV, lidos por cima dos do PHA) e `20261105-synthagent_revised_draft_v6 - jpv sections 3 - 4.pdf` (37 comentários). O JPV não chegou aos resultados.

Estado face à v6 atual: **R** = já resolvido na v6; **P** = correção pequena; **G** = trabalho maior; **C** = em conflito com um pedido do PHA; **N** = proposta de não alterar (com razão).

## v5 (resumo, introdução, antiga Secção III)

| # | Local | Comentário (resumido) | Estado | Proposta |
|---|---|---|---|---|
| 1 | Resumo | Os LLMs também não "aprendem com os registos disponíveis"? | R | O resumo diz agora que os geradores estabelecidos aprendem só com os registos disponíveis e que o LLM não é treinado nos dados-alvo. |
| 2 | Resumo | "Que synthetic dataset?" — mencionado sem explicar de onde vem | R | Resumo reescrito (a pedido do PHA). |
| 3–5 | Resumo | Evitar "them", "it": cria ambiguidade; usar os nomes | R/P | Frases corrigidas; fazer uma passagem final ao artigo todo à procura de pronomes ambíguos. |
| 6 | Resumo | "The control layer then asks the model to…" | R | Frase reescrita. |
| 7 | Resumo | 6 datasets não chegam para argumentar generalidade | N | Sem tempo para novos datasets; já está nas limitações. Não afirmar generalidade. |
| 8 | Resumo | Primeira menção a classificadores sem dizer como é feita a avaliação | P | Dizer no resumo que a utilidade é medida pelo F1 de dez classificadores treinados com e sem registos sintéticos. |
| 9 | Resumo | "cost" em termos de quê? | P | "custo em tokens" no resumo. |
| 10 | Resumo | Qual é o número pedido de registos e como foi escolhido? Se for baixo, todos cumprem | P | Dizer que N = tamanho do treino (o treino é duplicado) e que os registos são pedidos em lotes de 30, por isso a falha da prompting direta não se deve a um N excessivo. |
| 11 | Resumo | "number of records is low" é subjetivo | R | Definido com Riley et al. |
| 12 | Intro | SDG mitiga small data, mas também falta de dados em geral | P | Uma palavra: "mitigate small data and, more generally, a lack of data". |
| 13 | Intro | Que parte do Cover & Thomas suporta a afirmação? | P | Citar a desigualdade do processamento de dados (Sec. 2.8). |
| 14 | Intro | Não treinar no dataset-alvo não reduz a accuracy face a quem treina? | N | É respondido pela RQ4 (comparação com SMOTE, CTGAN, TVAE). Pode-se apontar para lá. |
| 15 | Intro | "free text" só se a prompt o pedir | P | Reformular: "returns text that has to be parsed, and nothing guarantees that it is a dataset". |
| 16 | Intro | "Só" um problema de validação? O desafio está na validação ou na produção? | G | Ver tema A (contribuição central). |
| 17 | Intro | A reparação, correção e o pedido exato dos registos em falta seriam a contribuição mais relevante, mas dizes pouco sobre isso | G | Ver tema A. |
| 18 | Intro | Responder às RQs na intro não é habitual; pôr só um resumo dos insights | C | O PHA pediu explicitamente as conclusões na intro com (RQx). A versão atual já é um resumo curto; manter. |
| 19 | III | "a system can guarantee them" — them o quê? | R | Secção III removida. |
| 20 | III | "faithful" é vago | P | Ainda aparece na RQ2 e na discussão: substituir por "high-fidelity"/"preserve the distribution". |
| 21 | III | Novo parágrafo em "Formally, let…" | R | Secção removida. |
| 22 | III | "active rules R" — regras de quê? | P | Verificar que a Eq. 1 já não usa R sem definição. |
| 23 | III | Os pesos podiam ser definidos de forma abstrata | N | A formulação multiobjetivo atual não usa pesos; explicar numa frase porquê. |
| 24–25 | III | A função objetivo usa L_fid, L_priv, ΔU, κ sem as definir | P | Definir cada componente numa frase, com a medida da Secção IV. |
| 26 | III | Formalismo demais para a introdução | C | O PHA pediu a formulação na introdução. Ver tema C. |

## v6 (Secções 3 e 4)

| # | Local | Comentário (resumido) | Estado | Proposta |
|---|---|---|---|---|
| 27 | III | Eq. (1) fica muito longe, na intro; intro ficou pesada (parágrafo de uma coluna inteira) | C | Ver tema C. |
| 28–29 | III | Nomes diferentes da Figura 2 (passos 3 e 4) | P | Alinhar nomes do texto e da figura (ver tema B). |
| 30 | III | Falta descrever a saída (último bloco da Figura 2) | P | Uma frase: tabela sintética G e log de auditoria. |
| 31 | III/Tab. II | Parâmetros não aparecem na Figura 2; justificações incompreensíveis antes de explicar as etapas; porque não há estudo de sensibilidade? | G | Ver tema D. |
| 32 | III-A | B_t, E_t aparecem sem explicação | P | Introduzir cada símbolo quando é usado. |
| 33 | III | Contar uma história: visão geral primeiro, depois detalhes | G | Ver tema B. |
| 34–36 | III-A | Pequenas alterações: "from batch Y_t", "by K_{V_S}", "the parser log" | P | Adotar. |
| 37 | III-A | Título → "Controller state" (coerente com a Figura 2) | P | Adotar. |
| 38 | III-B | "no dataset receives more prompt engineering" — o core é prompt engineering? Ou o parsing/validação (descrito superficialmente)? | G | Ver tema A. |
| 39 | III-B | "context selector" = "context budgeter" da Figura 2? Usar os mesmos termos | P | Alinhar com a figura. |
| 40 | III-B | Formalização demais; apresentar como algoritmo | G | Ver tema B. |
| 41 | III-B | Começar pelo caso de uso e pelo algoritmo, depois detalhar as componentes | G | Ver tema B. |
| 42 | III-B | Tabela III é sobre o que o modelo recebe em geral; porquê aqui? | P | Movê-la para a visão geral (ou para "Prompts", ver tema A). |
| 43 | III-B | Corta o fluxo; saltos frequentes entre partes do artigo | G | Ver tema B. |
| 44 | III-B | ρ = 20% na especificação: devia ser genérica, valores escolhidos por estudo de sensibilidade | G | Ver tema D. |
| 45 | III | Se a contribuição é a arquitetura como um todo, é insuficiente; descrever a contribuição principal com mais profundidade | G | Ver tema A. |
| 46 | III | Não descreve como são feitas as prompts; a qualidade não depende muito delas? | G | Ver tema A. |
| 47 | III-C | b = 30 arbitrário, sem justificação | G | Ver tema D. |
| 48 | III-C | Parsing, validação e correção (passos 2–4) descritos de forma tão sumária? | G | Ver tema A. |
| 49 | III-E | A implantação entre instalações não está ao mesmo nível das outras subsecções | P | Passar a parágrafo ou secção própria no fim da Secção III, ou para a discussão. |
| 50–53 | Alg. 1 | Explicar A←∅, ωW; demasiada nomenclatura; dar nomes às variáveis ("parse the batch Y", não Y_t?) | P | Comentários e nomes no algoritmo. |
| 54 | IV | "of TBD-especifica" | P | Completar a frase. |
| 55–56 | IV | "harder" → "a challenge for the classifier" | P | Adotar. |
| 57 | IV | "Using several families…" — good! | — | — |
| 58 | IV | Enumerar os três cenários: (1) só reais, (2) só sintéticos, (3) reais + sintéticos | P | Adotar. |
| 59 | IV | Não percebe porque o XGBoost falha com as classes {1,2} | P | Explicar: o XGBoost exige classes codificadas de 0 a K−1. |
| 60 | IV | Não se percebe na Tabela V quais são as 22 configurações | P | Uma coluna com o número de configurações por linha e o total. |
| 61–62 | IV-D | Medidas como uma lista em texto corrido; dar contexto primeiro; texto fluido, não mecanizado | P | Frase de abertura que explique o que se mede e porquê, e ligar as medidas às RQs. |

## Temas maiores

**A. Contribuição central.** O JPV pergunta várias vezes (16, 17, 38, 45, 46, 48) qual é o núcleo técnico: a prompt, a validação, a correção, o completion loop ou a arquitetura. A Secção III gasta mais espaço em formalismo do estado e do contexto do que no parsing, na validação, na correção e no completion loop, que são o que produz os resultados da RQ1. Proposta: dizer explicitamente que a contribuição é a garantia de completude e conformidade (exatamente N registos com as colunas e tipos do esquema) e descrever com mais detalhe os passos 2–4 e a construção das prompts.

**B. Estrutura da Secção III.** Começar pela visão geral (Figura 2 e Algoritmo 1, com nomes iguais), depois detalhar as componentes pela ordem do algoritmo, com menos notação.

**C. Formulação do problema na introdução.** O PHA pediu-a na introdução (v5, #28); o JPV acha-a pesada e longe da Secção III. Compromisso possível: manter a Eq. 1 na introdução (pedido do PHA), partir o parágrafo grande em dois, e lembrar a formulação numa frase no início da Secção III.

**D. Parâmetros fixos e estudo de sensibilidade.** O JPV pede valores genéricos na especificação e escolhidos por um estudo de sensibilidade. Opções: (1) só texto: tirar os valores da especificação e justificar a Tabela II depois de explicar as etapas; (2) correr um pequeno estudo de sensibilidade com GPT-4o mini para ρ e b.
