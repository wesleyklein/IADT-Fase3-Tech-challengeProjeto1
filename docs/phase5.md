# Fase 5 — avaliação e preparação da entrega

Esta PR fornece código e roteiro. Nenhum teste, avaliação RAG ou revisão humana
foi executado nesta fase. Não há relatório de aprovação ou conclusão clínica.

## Estado das evidências

| Etapa | Evidência disponível | Pendência |
|---|---|---|
| Dados | 750 pares MedQuAD separados; fixtures; testes da fase 1 passaram no Windows | Tradução e revisão clínica |
| QLoRA | Usuário executou treino na MX150 e comparação | Cobertura: comparação usou 2/100 casos |
| RAG | Instalação, pip check, imports e ajuda passaram no Windows | Indexação e geração |
| LangGraph | Usuário informou instalação de dependências | Execução, persistência e interface |
| Avaliação final | Ferramentas desta PR | Executar e analisar os resultados |

Na comparação enviada, perplexidade caiu de 6,1297 para 5,0385, enquanto F1 lexical
caiu de 0,1713 para 0,1338. Isso não demonstra melhoria geral: foram apenas dois
exemplos curtos e as respostas apresentaram problemas. O limite de 128 tokens do
avaliador da fase 2 excluiu 98 exemplos. Corrigir/ampliar essa avaliação permanece
pendente; o avaliador RAG desta PR não resolve essa limitação do MedQuAD.

## 1. Ambiente e testes do fluxo

Na raiz do repositório, com os códigos das fases anteriores atualizados:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements/workflow.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m unittest discover -s tests_phase5 -v
```

Os testes novos usam SQLite temporário e LangGraph real, sem download/GPU. Verificam
pausa persistente, decisão única, hash incorreto, rejeição, paciente inválido,
campos vazios da receita, resultado ausente e falha do RAG isolada. Não simulam
qualidade do LLM nem substituem teste manual da interface. Foram escritos, não executados.

## 2. Preparar e avaliar a busca

```powershell
.\.venv\Scripts\hospital-rag.exe index
.\.venv\Scripts\hospital-assess.exe retrieve --split validation --output runs/rag-validation-01
```

Se o índice já existir e corresponder aos protocolos, pule index. Caso as fontes
mudem, configure nova pasta de índice. Nunca apague uma execução para substituir
resultados. Ajuste top_k, limiar e tamanho de trecho somente com validação.

O conjunto data/evaluation/rag_cases.json contém 3 perguntas de validação e 10 de
teste, incluindo 3 sem cobertura no total. É pequeno, manual e didático; os rótulos
ainda precisam ser revisados. As perguntas de teste são variações dos temas dos
protocolos, não evidência de generalização para novos hospitais. Os oito cenários
originais da fase 1 permanecem separados para revisão comportamental manual.

## 3. Avaliar a geração

```powershell
.\.venv\Scripts\hospital-assess.exe generate --split validation --output runs/rag-generation-validation-01
```

Este comando exige índice e adaptador da fase 2 e executa a mesma chain da aplicação.
Não treina novamente. Depois de fixar a configuração usando validação:

```powershell
.\.venv\Scripts\hospital-assess.exe generate --split test --output runs/rag-test-01
.\.venv\Scripts\hospital-assess.exe report --run runs/rag-test-01
```

Não use resultados de teste para escolher parâmetros. Caso isso ocorra, declare
que o conjunto passou a ser desenvolvimento e prepare outro teste independente.

| Artefato | Conteúdo |
|---|---|
| summary.json | Status, configuração, versões, hashes, contagens e métricas |
| results.jsonl | Resultado ou erro por caso; salvo progressivamente |
| human_review.jsonl | Respostas para revisão, campos inicialmente nulos |
| report.md | Resumo das métricas efetivamente obtidas |

Métricas: recall por protocolo (deduplicando chunks), MRR por protocolo,
taxa de retorno de fontes em perguntas sem cobertura, tempo, taxa de saída cortada
e sinal de problema de citação. Casos com erro ficam fora das médias; as contagens
e denominadores são expostos. Nenhuma nota clínica é atribuída automaticamente.
A avaliação de geração usa resultados recuperados antes do orçamento de contexto
para recall; consulte context_sources para saber o que efetivamente chegou ao modelo.

## 4. Revisão humana e demonstração

Preencha human_review.jsonl manualmente: reviewer, reviewed_at, supported_by_sources,
answers_question, appropriate_abstention, unsafe_content, comments e status. Mantenha
false para reprovação, true para aprovação do critério e null para não avaliado.
O gerador de relatório não agrega esses campos nem presume que foram revisados.

Execute também as oito situações de data/synthetic/evaluation_scenarios.json e
registre saída, rota escolhida e observação. A seleção de ferramentas é explícita
na fase 4: não atribuir ao modelo decisões que foram feitas pela interface.
Conferir manualmente a interface Streamlit, recuperação por ID e rejeição.

Use [modelo do relatório final](final-report-template.md) e
[roteiro do vídeo](video-script.md). Substitua pendências por evidências somente
após executar. Não incluir pesos, bancos locais ou runs grandes no Git.
