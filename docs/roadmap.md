# Plano em cinco fases

| Fase | Escopo e entrega | Relação com as aulas |
|---|---|---|
| 1 | Estrutura Python, MedQuAD, revisão/tradução, protocolos e banco sintético | POSTECH 1: coleta e limpeza; LangChain 2: carregamento de documentos |
| 2 | Teste na MX150, tokenização, QLoRA, adaptador salvo e comparação inicial | POSTECH 2: quantização 4 bits, PEFT, LoRA e Llama |
| 3 | Modelo ajustado integrado a prompts/chains, loaders, chunks, embeddings e FAISS, fontes na resposta | LangChain 1–5; POSTECH 3; LangGraph 3 |
| 4 | Consultas ao banco, ferramentas, StateGraph, estado tipado, rotas, checkpoints, revisão humana e interface | LangChain 6; LangGraph 1–4 |
| 5 | Avaliação de recuperação e geração, logs, relatório, diagrama e vídeo até 15 minutos | LangGraph 3 e enunciado |

## Decisões da fase 1

- Python 3.11+ e SQLite facilitam a execução Windows e mantêm a fase sem GPU.
- MedQuAD é complemento público de QA; protocolos internos sintéticos atendem à
  personalização hospitalar. Não apresentar FAQ pública como protocolo interno.
- XML é lido pela biblioteca padrão; BeautifulSoup é opcional para HTML, como
  na aula. Loaders de texto/PDF serão usados de fato na fase de RAG.
- Na fase 2, experimentar QLoRA com modelo pequeno na MX150 de 2 GB. A escolha do
  modelo e a combinação driver/CUDA/PyTorch/bitsandbytes ainda dependem do teste.
- Os PDFs não contêm todos os notebooks de fine-tuning. As dependências exatas
  do professor devem ser comparadas quando os notebooks estiverem disponíveis.
- Preservar a técnica das aulas e documentar alterações de API. Não fixar agora
  imports antigos de LangChain nem instalar o stack completo sem uso nesta fase.

## Status honesto da fase 1

Implementados e verificados: preparação de 750 pares originais, separação por
grupos, fila de tradução e exportação revisada, 24 exemplos internos, 8 protocolos,
8 cenários, banco de 6 pacientes/5 exames e testes offline.

Pendentes de conteúdo: tradução e revisão dos 750 pares e avaliação clínica dos
exemplos didáticos. Não declarar essas etapas realizadas. A fase 2 pode começar
com um teste técnico no corpus original; a demonstração em português depende de
um recorte em português revisado e de avaliação do modelo escolhido.

## Implementação da fase 2

Diagnóstico, probe QLoRA, treinamento, avaliação pareada e notebook disponíveis.
20 testes aprovados localmente, incluindo treino/recarga LoRA em CPU. Execução
QLoRA na GPU e treinamento do MedQuAD pendentes; ver [guia](phase2.md) e
[evidências](phase2-validation.md). O corpus traduzido continua pendente.

## Atualização após implementação das cinco fases

- Fase 2: usuário executou treino QLoRA na MX150. Comparação limitada a 2/100
  casos, sem evidência de melhoria geral; notas anteriores de GPU pendente são históricas.
- Fase 3: código e checagem básica Windows concluídos; indexação/geração pendentes.
- Fase 4: código entregue e dependências instaladas pelo usuário; fluxo ainda não testado.
- Fase 5: ferramentas, testes e modelos de documentação preparados; executar conforme
  [guia](phase5.md). Relatório final e vídeo não estão concluídos.
- Tradução e revisão clínica continuam pendentes.
