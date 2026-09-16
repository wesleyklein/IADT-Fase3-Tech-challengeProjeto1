# Comparação controlada do RAG

A recuperação v2 encontrou os protocolos esperados na pequena validação, mas o
modelo ainda recusou a pergunta de identificação e omitiu citações. Este experimento
compara base/adaptador × prompt atual/curto. Não escolhe automaticamente um vencedor.

Após integrar a PR, no PowerShell, na raiz do projeto:

```powershell
git pull
.\.venv\Scripts\python.exe -m unittest discover -s tests_phase5 -v
.\.venv\Scripts\hospital-assess.exe compare --config config/rag-v2.json --split validation --output runs/rag-comparison-v2-01
```

Use a configuração e o índice v2 já construídos e o treino concluído em
`runs/qlora-mx150-01`. Não é necessário reindexar ou treinar novamente. Para repetir,
escolha outra pasta de saída: resultados existentes não são sobrescritos.

A busca ocorre uma vez por pergunta. O contexto selecionado precisa caber nos dois
prompts e é reutilizado nas quatro combinações. O modelo é carregado uma vez; o
context manager PEFT desliga o adaptador para a base e o restaura ao sair. A precisão,
os pesos base e a geração determinística permanecem iguais entre as variantes.
A configuração padrão do aplicativo não muda.

Arquivos gerados:
- `summary.json`: métricas por combinação, prompts, configuração, versões e hashes.
- `results.jsonl`: respostas, fontes, hash do contexto, variante e tempo por caso.
- `human_review.jsonl`: cópia dos resultados com campos de revisão pendentes.

Confira `status` e eventuais erros antes de interpretar as métricas. Com os três
casos atuais e a mesma cobertura, são 12 registros: oito gerações e quatro respostas
automáticas sem contexto. As taxas de citação e truncamento consideram somente
chamadas concluídas do modelo; abstinências automáticas têm contador separado.
Isso também foi corrigido no comando `generate`. Não compare diretamente a antiga
taxa de 2/3 com a nova de 2/2: os denominadores são diferentes.

Compare cada pergunta nas quatro variantes: responde à pergunta? A fonte sustenta
o texto? A citação existe e corresponde à regra usada? No caso sem cobertura, a
abstinência é adequada? Uma citação válida não comprova pertinência ou sustentação.
Preserve a revisão humana: o programa não preenche aprovação automaticamente.

Use `summary.json` diretamente para esta comparação (o comando `report` é destinado
às avaliações retrieve/generate). Tempos incluem overhead e aquecimento; não são um
benchmark de desempenho. Três casos são insuficientes para concluir qualidade geral.
Escolha ajustes na validação e somente depois execute a avaliação separada de teste.
Os testes automatizados usam modelo simulado: a comparação real depende do adaptador
local e deve ser executada no computador onde o treinamento foi salvo.
