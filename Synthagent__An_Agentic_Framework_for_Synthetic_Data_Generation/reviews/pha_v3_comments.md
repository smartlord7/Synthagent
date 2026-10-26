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
