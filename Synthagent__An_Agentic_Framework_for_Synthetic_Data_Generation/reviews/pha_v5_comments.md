# Comentários do PHA ao draft v5 e alterações na v6

| # | Local | Comentário (resumido) | Alteração na v6 |
|---|---|---|---|
| 1 | Resumo: "for example because a disease is rare…" | A complexidade da task não é o que leva a small datasets; nomear a task. | A tarefa vem primeiro ("Prediction tasks in healthcare, such as diagnosing a disease or anticipating an adverse outcome"), seguida das causas dos datasets pequenos. |
| 2 | "these patterns" | → "this issue" | Adotado. |
| 3 | "because they bring" | → "by adding" | Adotado. |
| 4 | "The control layer infers" | → "inferring" | Adotado ("Inferring the columns and their types…"). |
| 5 | Resultados no resumo | Não se percebe a mais-valia; custo maior e F1 que reduz. | Resultados reescritos pela positiva: dataset utilizável sempre entregue (a prompting direta nunca o fez), aumento de custo desprezável, maior ganho de F1 entre os geradores, e evitadas as perdas de até 10,6 pontos dos outros geradores. |
| 6 | "prediction tasks" | Então estás focado nesta problemática. | O foco é explícito: título, resumo, keywords e introdução ("This paper focuses on SDG for prediction tasks on small healthcare datasets"). |
| 7 | "In this paper" | → "In our work" | Adotado. |
| 8 | "what the learning task requires" | Adivinhar este número era magia. | Small data passa a ser "too low to train a reliable prediction model", com Riley et al. referido como fator de dependência e não como número exato. |
| 9 | "They can only recombine" | → "recombining patterns that already exist" | Adotado. |
| 10 | "a different source of information" | → "a new perspective to address such issue" | Adotado. |
| 11 | "requested by the user" | Como sabe o user quantos records precisa? | N é apresentado como o número de registos sintéticos com que qualquer método de augmentation é configurado (proporção do treino, como no SMOTE). Na avaliação, N = tamanho do treino, ou seja, o treino é duplicado. O problema é o gerador devolver silenciosamente menos registos ou registos mal formados. |
| 12 | "a known cost" | Que cost? | Definido: custo monetário de cada pedido a um LLM comercial, proporcional aos tokens processados. |
| 13 | Single vs two runs | Argumento forte. | Argumento concreto: com 40 registos de teste, um único registo mal classificado muda a accuracy em 2,5 pontos; a aleatoriedade da divisão e da amostragem do LLM faz duas execuções diferirem tanto quanto as diferenças comparadas. |
| 14 | Contribuição "control layer" | O sistema não é só a control layer? | O SYNTHAGENT é apresentado como LLM + control layer; as contribuições deixam de vender a control layer separadamente. |
| 15 | Antes das RQs | Deveria existir mais texto. | Frase de objetivo antes das RQs. |
| 16 | Respostas às RQs | "Main conclusions" com a RQ no fim. | "The main conclusions are the following", cada conclusão termina com (RQx). |
| 17 | "ignore what the features mean" | O que significa? | Explicado: tratam cada coluna como um número anónimo, sem saber que é a idade ou o colesterol. |
| 18 | "estimate many parameters from few records" | Estranho: poucos dados e mais parâmetros? | Reformulado como limitação: estes modelos têm muitos parâmetros por construção, que poucas centenas de registos não estimam de forma fiável. |
| 19 | "valid dataset of the requested size, cost" | Como resolves o valid output? | "Valid" definido como número configurado de registos com as colunas e tipos corretos; a Secção III explica como é garantido (parser, validador, correções, completion loop). |
| 20 | Runs em II-B | O que já disse. | Remete para o argumento da introdução. |
| 21 | II-C | Não sei se é bem conseguida. | Reduzida a "Validation of Generated Data": um parágrafo focado no princípio de verificar dados gerados contra restrições. |
| 22 | "We now state the problem…" | Frase de aula. | Removida com a secção. |
| 23 | N | Necessidade do N. | Ver 11. |
| 24 | R2 "new?" | — | Removidos todos os rótulos R. |
| 25 | R4 | Porque tem que ser? | A fidelidade é justificada como objetivo de avaliação padrão (Synthcity, SynthEval) e explicada na formulação. |
| 26 | R5 | Vais pela similaridade e descartas? | Clarificado nas limitações: o SYNTHAGENT não filtra registos próximos dos reais; a privacidade é medida com um ataque, não imposta. Filtros de privacidade ficam como trabalho futuro. |
| 27 | "improves, or at least does not degrade" | Não faz sentido. | O objetivo passa a ser maximizar o ganho dos classificadores; a ausência de perdas grandes aparece apenas como resultado (robustez). |
| 28 | Secção III | Não deveria existir; formulação na introdução. | Secção III removida; a formulação matemática (Eq. 1) está na introdução, no parágrafo do problema. |
| 29 | Tabela I | Deveria aparecer no sítio. | Colocada no início da Secção II. |
| 30 | ✓ vs (✓) | Faz sentido diferenciar? | Só ✓/×; as nuances passaram para a legenda. |
| 31 | "enter the synthetic dataset" | Sintético ou original? | Clarificado: dataset sintético; o dataset real nunca é modificado. |
| 32 | 20% de contexto | Quem decidiu? Pode ser 2–4 instâncias. | Os números reais são dados (31 a 123 registos) e justificados na nova tabela de parâmetros (Tabela II). |
| 33 | Percentagens e valores ad hoc | Tudo parece ad hoc. | Nova tabela de parâmetros (Tabela II) com todos os parâmetros fixos e a razão de cada valor; declarado que foram fixados antes das experiências e não afinados. Detalhes de implementação (códigos HTTP, caracteres por token) removidos. |
| 34 | "30" | Porquê 30? | Justificado na Tabela II. |
| 35 | Temperatura 0,1, tolerâncias | Idem. | Tabela II; as regras opcionais são referidas sem números. |
| 36 | Algoritmo 1 | Pseudo-código com n thresholds fixos. | Reescrito em passos de alto nível, com os parâmetros nomeados (ρ, ω, b, c, t_max, u_max) e remetidos para a Tabela II. |
| 37 | Healthcare | Nunca dizes que o método é para healthcare. | Explícito no título, resumo, keywords, introdução e contribuições. |
| 38 | Federado | Preparar o leitor bem antes. | Introduzido no 1.º parágrafo (cada instalação aprende com os seus dados), no 4.º (cada instalação corre o sistema localmente) e nas contribuições (padrão de implementação); a Secção III-E remete para a introdução. |
| 39 | "tudo é R" | Confuso. | Rótulos R removidos de todo o artigo. |
| 40 | "We use healthcare data because…" | Era na intro; aqui só "escolhemos X datasets públicos". | Motivação na introdução; aqui fica "Six public classification datasets were selected". |
| 41 | 14 configurações | O que são? O que é "completed"? | Nova Tabela V com as 14 configurações por grupo e finalidade; "completed" definido como os seis datasets × dez repetições. |
| 42 | "reference Synthagent" | O que é esta baseline? | Definida: SYNTHAGENT com as definições por omissão e o GPT-4o mini; todas as outras configurações diferem dela num único aspeto. |
| 43 | Leitura dos resultados | Muito difícil de ler. | Cada RQ abre com a resposta numa frase; o texto foi reduzido para cerca de metade e os números ficam nas tabelas. |
| 44 | Agradecimentos: uso de LLMs | Adicionar texto. | Não adicionado, por decisão do autor. |
