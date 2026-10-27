# Comentários do PHA ao draft v3 (resumo e introdução)

## Chat

- "O artigo continua a apresentar os mesmos problemas." Leu o resumo e a introdução.
- "Não estás a melhorar na escrita... continuas a fazer os mesmos erros."
- "Não quero que escrevas, escrevas e escrevas sem um fio condutor."
- Recomendação: a introdução deve ter **no máximo 1 página**, com **um parágrafo para cada ponto**:
  1. contexto
  2. problemática
  3. trabalhos na área e suas principais limitações
  4. o teu trabalho e porque é importante
  5. como conseguiste atingir os teus objetivos, com resultados
  6. estrutura do documento

## PDF: Resumo (p. 1)

| Texto destacado | Comentário |
|---|---|
| "a few hundred patient records, fewer than the task requires, that cannot be pooled with those of other hospitals" | Estás a assumir (1) que small data é poucas instâncias e (2) que tens vários hospitais envolvidos. Nenhum destes pontos é uma verdade absoluta. |
| "information that the records lack" | Não sei o que isto significa. |
| "Large language models (LLMs) can, because they bring knowledge from pre-training…" | Não percebi o "can". |
| "Their raw output, however, is not a dataset. Asked directly for a fixed number of records… none of 60 runs, delivering between 57% and 103%…" | Percebi "0". |
| "…F1 of a dataset by more than 0.6 points, whereas the other generators lose up to 10.6 points" | Ninguém percebe isto. |
| "No generator improves every dataset, a limit that follows from how much information each generator can contribute." | O que queres dizer com isto? Que o que estás a tentar fazer é impossível? |
| Nota na margem, junto às LLMs no resumo | Falta-te contexto, sobretudo relacionado com trabalhos na literatura. Além disso, quando colocas as LLMs, deixa-se de perceber. |
| Index Terms (nota) | Uma lista de keywords deve ter idealmente 3–4 no máximo; pelo menos "data augmentation" poderia sair. |

## PDF: Introdução (pp. 1–2)

| Texto destacado | Comentário |
|---|---|
| "…most clinical prediction tasks are not large." | "suffer from the lack of data?" |
| "a rare disease may have only a few dozen cases;" | Queres colocar a parte das doenças raras no início da frase? Se calhar ficaria melhor. |
| "In such settings" | "In such settings" não, porque acabaste de falar de privacidade, e isso nada tem a ver com small data. |
| "What counts as too few depends on the task." | Frase literária de romance. |
| "Three hundred patients can suffice to predict a common outcome… and fall far short for a rare outcome…" | É uma afirmação estranha: seja em que contexto for, 300 doentes serão sempre insuficientes. |
| "Our recent survey [1] makes this intuition measurable" | Não te podes referir a um trabalho que nem sequer está publicado (nem em vias disso). |
| Nota na margem, mesma frase | O que é "intuition measurable"? |
| Parágrafo sobre sample complexity, effective sample size e smallness index | Esta informação, que vem da referência ao survey, surge sem qualquer contexto e fica muito difícil perceber porque é que precisamos de a ler. |
| Nota no fim desse parágrafo (p. 1, coluna direita) | Como leitor, chego a este ponto e ainda não percebi (1) o problema que vais tratar, (2) o que existe e o que não existe, (3) o que propões e porque é novel. |
| "How can a model be learned from such data? The survey classifies the possible remedies…" (nota) | Mais uma vez, "the survey classifies…" não pode ser. |
| "Large language models (LLMs) are different, and this is why we turn to them." | Outra frase de romance. |
| Nota junto a "…none of 60 runs, delivering on average 87.6%…" (p. 2) | Já estás a falar em resultados sem teres sequer explicado a problemática, etc. |
| "This is the gap the paper addresses," | Mas que gap, e onde é que esse gap existe? |
| Nota junto a "…satisfies R1–R3 by construction…" (p. 2) | Só mais um exemplo: falas em R1 e R3. Onde é que estão? Mais para a frente… |
| Nota junto a "The third is an empirical study…" (p. 2) | E o empirical study não é o system? Ou o system não inclui o estudo? |

## Como foram endereçados (v4)

| Comentário | Alteração na v4 |
|---|---|
| Introdução com no máximo 1 página, um parágrafo por ponto | A introdução tem agora exatamente seis parágrafos, por esta ordem: contexto, problemática, trabalhos na área e limitações, o nosso trabalho e porque é importante, resultados, estrutura do documento. Ocupa cerca de uma página. |
| Small data = poucas instâncias / vários hospitais | Small data é definido como um número de registos baixo **em relação ao que a tarefa exige** (número de features, de classes e frequência do outcome). Doenças raras, estudos num único hospital e custo de rotulagem aparecem como exemplos, não como pressupostos. A partilha entre instituições é apenas uma possível mitigação, "frequentemente restringida". |
| "information that the records lack", "can", "0", "ninguém percebe", "no generator improves every dataset" | O resumo foi reescrito por esta ordem: contexto, problema, limitação dos geradores, porquê LLMs, falha das LLMs com números concretos (88% em média, 10 respostas inválidas), proposta e resultados. Os resultados aparecem em termos de F1-score dos classificadores. A frase sobre "no generator" foi removida. |
| Keywords 3–4 | Passaram a quatro: small data, synthetic data generation, large language models, tabular data. |
| "not large" → "suffer from the lack of data" | Adotado: "Many clinical prediction tasks suffer from a lack of data." |
| Doenças raras no início da frase | Adotado. |
| "In such settings" a seguir à privacidade | Removido; a privacidade surge apenas como restrição à partilha de dados. |
| Frases de romance ("What counts as too few…", "this is why we turn to them", "intuition measurable") | Removidas. |
| "300 doentes podem ser suficientes" | Removido. |
| Citar o survey não publicado | Todas as referências ao survey foram removidas. A definição de small data usa o critério publicado de Riley et al. O argumento de que os geradores ajustados aos dados não acrescentam informação usa a desigualdade de processamento de dados (Cover & Thomas). A tabela de datasets mostra o rácio entre o tamanho mínimo recomendado por Riley et al. e o tamanho de treino. |
| Informação do survey sem contexto | O parágrafo sobre sample complexity, effective sample size e smallness index foi eliminado. |
| "Ainda não percebi o problema, o que existe e o que é novel" | Cada um destes pontos tem agora o seu parágrafo: 2.º (problema), 3.º (trabalhos e limitações) e 4.º (proposta e porque é importante). |
| "The survey classifies…" | Removido. |
| Resultados antes da problemática | Os números só aparecem no 5.º parágrafo; no resumo, aparecem depois do problema. |
| "Que gap e onde?" | O 3.º parágrafo diz explicitamente o que falta em cada linha de trabalho. |
| R1 e R3 usados antes de definidos | A introdução já não usa R1–R6. Os requisitos são definidos explicitamente no início da Secção III, e a Tabela I usa nomes em texto ("Exact size", "Schema validated", "Cost measured"). |
| "O empirical study não é o system?" | Passam a ser duas contribuições: o sistema e a metodologia de avaliação. O estudo é a aplicação dessa metodologia. |
| (revisão final) Termos usados antes de definidos | RQ1–RQ4 são agora enunciadas explicitamente no 5.º parágrafo da introdução, antes de serem usadas nos resultados. |
| (revisão final) "Information that the records lack" (reformulação) | A ideia passou a ser explicada de forma concreta: os geradores ajustados aos dados "só podem recombinar padrões já presentes nesses registos, por exemplo interpolando entre dois doentes existentes"; as LLMs trazem conhecimento geral sobre variáveis clínicas (intervalos plausíveis, relações típicas). |
| (revisão final) Porque é importante | O 4.º parágrafo diz porque importa: o conhecimento das LLMs torna-se utilizável onde os geradores estabelecidos são mais fracos, e o conjunto de dados resultante pode ser auditado como qualquer outra fonte de dados. |
| (revisão final) Resumo | Sem jargão ("valid JSON" e "tokens" foram substituídos) e com uma frase final de implicação. |
