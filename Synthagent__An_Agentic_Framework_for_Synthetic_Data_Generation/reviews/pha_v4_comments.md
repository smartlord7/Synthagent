# Comentários do PHA ao draft v4 e alterações na v5

| # | Local / texto | Comentário do PHA | Alteração na v5 |
|---|---|---|---|
| 1 | Resumo: "enlarge such datasets" | Substituir por "be a solution for such contexts" | Adotado. |
| 2 | Resumo: "the patterns those records already contain" | "patterns that already exist in the data"; como é que o teu método vai lidar com isto? | Adotada a formulação; a frase seguinte explica como o método lida com isso: as LLMs vão além desses padrões porque trazem conhecimento do domínio adquirido no pré-treino. |
| 3 | Resumo: "general knowledge" | "Contextual? Não sei se gosto de 'general'." | Passou a "domain knowledge". |
| 4 | Resumo: frase com 88% e 10 respostas | Frase de referência com percentagens; no resumo deve ser genérica. | Passou a genérica: "When prompted directly, LLMs often return fewer records than requested, malformed records or values of the wrong type". |
| 5 | Resumo: "This paper" | "To address such issue"; o problema tem de ficar claro antes. | "To address this issue", depois de o problema ficar descrito na frase anterior. |
| 6 | Resumo: schema, invalid records, missing ones | Demasiados conceitos: quem define o schema? O que são invalid records? Porque haveria missing? | Reescrito em termos concretos: o sistema infere nomes e tipos das colunas a partir dos dados reais, verifica cada resposta, pede a correção dos valores que os violam, e continua a pedir registos até atingir o número definido pelo utilizador. Que as LLMs devolvem menos registos do que o pedido é dito na frase anterior. |
| 7 | Resumo: "requested size, 99.7% to 100%" | Dizer qual o objetivo do "requested size"; 100% deixa de pé atrás. | Removidos os 99,7–100%. O tamanho é "o número definido pelo utilizador", e o resultado diz que a prompting direta nunca o atingiu. |
| 8 | Resumo: "never decreased … up to 10.6" | Porque decresce? | No resumo: "o único gerador cujos registos nunca baixaram o F1 de nenhum dataset mais de um ponto". Na introdução explica-se porque pode baixar: registos sintéticos podem prejudicar um classificador quando representam mal os dados reais. |
| 9 | Resumo (nota) | Estes resultados ainda estão estranhos. | Resultados reformulados: ganho médio de F1 com o melhor LLM, maior ganho entre os geradores, e pior caso explicado. |
| 10 | Introdução / resumo (nota) | Se o foco é healthcare, porque não também no resumo? | O resumo começa com "Prediction models in healthcare" e fala de "healthcare datasets". |
| 11 | Intro: "and how to determine whether such a dataset improves classifiers" | A melhoria da classificação é consequência e não deve estar colada ao objetivo. | O problema é obter o dataset, "so that it can be used to improve the performance of classifiers". |
| 12 | Intro: "The problem addressed in this paper is" | "One problem that stands …"; tirar "this paper", que serve para apresentar a abordagem. | Adotado: "One problem that stands out is therefore how to …". |
| 13 | Intro: "Existing work does not solve this problem." | "Currently, Deep …" | Adotado. |
| 14 | Intro: nome do sistema | Este nome não significa nada? | Explicado: SYNTHAGENT = synthetic data agent; como um agente LLM, alterna ações (pedidos ao LLM) e observações (verificações do output). |
| 15 | Intro: SMOTE, GAN, VAE | 3 siglas sem explicação; usar "pre-processing techniques (e.g. SMOTE), generative adversarial networks (e.g. CTGAN) …" | Siglas por extenso na introdução: oversampling techniques (SMOTE, Synthetic Minority Over-sampling Technique), CTGAN (conditional tabular generative adversarial network), TVAE (tabular variational autoencoder). O resumo usa só as categorias. |
| 16 | Intro: "matters because" | "point is important? crucial? since" | "This point is crucial since …". |
| 17 | Intro: "The paper makes two contributions:" | Parágrafo próprio: "The main contributions are:" com lista. | Parágrafo próprio com lista de três contribuições. |
| 18 | Intro: metodologia de avaliação como contribuição | São contribuições assim tão diferentes? Retirava isto tudo. | Removida a metodologia de avaliação como contribuição. As contribuições são o sistema, a control layer e a implementação open-source. |
| 19 | Intro: parágrafo de avaliação | "In the Synthagent evaluation, six public healthcare (uniformiza o termo!) datasets containing …" | Reescrito e uniformizado: "healthcare" em todo o artigo, sem "medical" nem "clinical". |
| 20 | Intro: "ten data splits" | O que significa isto? | "Each dataset was randomly divided ten times into a training and a test set". |
| 21 | Intro: "against direct prompting and SMOTE, CTGAN and TVAE" | São estratégias de geração e devem ser apresentadas como tal; classificadores só no fim. | Apresentados como "three established generation strategies"; os classificadores aparecem depois. |
| 22 | Intro: RQs | Parágrafo próprio: "Four main research questions emerged in this work" com (RQ1)…(RQ4). | Adotado. |
| 23 | Intro: RQ1 | "Is there any relationship between the control layer and the reliability of the generated datasets?" | Adotado literalmente. |
| 24 | Intro: RQs (nota) | Têm de ser perguntas; depois: 1) RQs, 2) setup construído para lhes responder, 3) resultados, com resposta sucinta a cada RQ. | As quatro RQs são perguntas. Seguem-se o setup ("To answer these questions, an experimental setup was built …") e uma resposta sucinta a cada uma. |
| 25 | Intro: resultados (nota) | Pensar melhor como ilustrar os resultados; o "decreased" é estranho. | Os resultados estão organizados por RQ. A perda é explicada antes de ser quantificada. |
| 26 | Secção II: título | "Background Knowledge" | Adotado. |
| 27 | Secção II: "To see where Synthagent fits, …" | Linguagem pouco formal: "In this section, first a review of … is performed, then …" | Reescrito nesse registo. |
| 28 | Secção II | Explicar novamente o que significa SDG. | "synthetic data generation (SDG)" redefinido no início da secção. |
| 29 | II-A: SMOTE vs redes bayesianas | Num caso descreves o funcionamento, no outro a aplicação. | A descrição é paralela: cada família e cada exemplo são descritos pelo modo como produzem registos. |
| 30 | II-B: "of these generators" | "of the previously mentioned generators were" | Adotado. |
| 31 | II-B: "naming the limitations makes clear …" | Não percebi. | Removido. |
| 32 | II-B: "D" | O que é "D"? | Removidos os símbolos da Secção II; D e S só aparecem a partir da Secção III, onde são definidos. |
| 33 | II-B (nota) | Uma imagem com as famílias e limitações seria mais interessante. | A Tabela I passou a tabela de largura total: família, como produz registos, principal limitação em small data e propriedades. |
| 34 | II-B (nota) | L1–L4 + D não é a forma mais simples de perceber as diferenças. | Removidos L1–L4. O texto percorre as famílias por ordem e remete para a Tabela I. |
| 35 | II-C: "control layer" | Ainda não percebi a que se refere; o leitor já devia controlar? | Definido na introdução ("a deterministic software component placed between the LLM and the dataset, which checks every response of the LLM before any of its records is accepted"); a II-C remete para essa definição. |
| 36 | III: D, S | Só aqui é que sei o que é D e S. | Já não aparecem antes da Secção III. |
| 37 | III: requisitos | Não percebi a relação com os requisitos; de onde vêm? São aceites pela comunidade ou "Sancho like"? | Fundamentados: R1–R3 são dimensões de qualidade de dados (completude, conformidade com o esquema, custo; Wang & Strong 1996; Batini et al. 2009); R4–R6 são as três dimensões de avaliação de dados sintéticos (fidelidade, privacidade, utilidade; Synthcity, SynthEval). A relação entre os grupos é explicada: sem R1–R2 os dados nem podem ser avaliados em R4–R6. |
| 38 | IV: "never trusting the LLM with it" | "the output of the LLM?" | "by checking the output of the LLM instead of relying on it". |
| 39 | IV: transição para a Figura 2 | A frase é um "tiro": vens do trust e de repente aparece a figura 2. | Transição explícita: o princípio é explicado e depois "The architecture that implements this principle is shown in Figure 2". |
