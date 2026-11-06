# Comentários da reunião com os orientadores (v7) e alterações na v8

| # | Comentário | Alteração na v8 |
|---|---|---|
| 1 | Demonstrar melhor a parte nova: parsing, validação e controlo, com mais detalhe | Secção III-D: os três passos escalam de barato a caro (parser local, um pedido de reparação, rondas de correção) e um exemplo completo ligado à Figura 1 (22 registos recuperados, coluna inventada removida, tipo corrigido, 8 registos pedidos pelo completion loop). |
| 2 | Provar que a melhoria se deve à nossa novidade e não ao LLM usado | Novo parágrafo em V-A: com o mesmo LLM, prompts e contexto, a control layer garante a fiabilidade (100% dos registos contra 86% da prompting direta e 78% dos structured outputs) sem custo em utilidade (diferença mediana de F1 +0,08 pp, Wilcoxon p = .92); os ganhos de utilidade vêm do LLM que a control layer torna utilizável, por isso crescem com LLMs mais fortes. Uma prova direta exigiria correr a prompting direta com um LLM forte (não feito). |
| 3, 5 | Retirar os 100% do dataset da tabela; referir só no texto | Coluna de registos entregues removida da Tabela IX; referido no texto de V-A e V-D. |
| 4 | Dizer que a abordagem é consistentemente (em %) melhor em vez de mostrar o pior caso | Pior caso e perdas removidos da Tabela IX; nova coluna com a % de datasets em que o F1 melhora. Os números não sustentam "consistentemente melhor": o SYNTHAGENT com Sonnet 4.6 melhora em 50% dos datasets (CLLM 70%, CART 80%, com ganhos menores); o texto diz isto. |
| 6, 8 | Gráfico de radar para os vários datasets | Nova Figura 5: variação de F1 por dataset para cinco geradores representativos. |
| 7 | Abstract: problema, lacuna da literatura, abordagem, x datasets com x modelos, resultados | Reescrito por esta ordem. Não diz "melhor em fidelidade", porque CART, cópula gaussiana, ARF e CTGAN têm fidelidade maior; diz o que os resultados mostram (único gerador LLM que entrega sempre o dataset; maior ganho mediano de F1 com o LLM mais forte). |
| 9 | Usar mediana com desvio padrão | Tabela IX: ΔAcc e ΔF1 como mediana ± desvio padrão sobre datasets. |
| 10 | Usar bold nas tabelas | Bold na Tabela VI (melhor ΔF1 por dataset) e revisto nas Tabelas VII, IX e X. |
| 11 | Explicar os casos em que a abordagem não é a melhor e discutir | Novo parágrafo na Discussão: LLM fraco (kidney disease), fidelidade máxima (árvores, com risco de privacidade) e estabilidade (filtro CLLM, com dataset incompleto); combinação com um filtro como extensão. |
