# Validação da fase 1

Executada em ambiente Linux, Python 3.12, nesta implementação:

- 11 testes offline aprovados (`python -m unittest discover -s tests -v`).
- Importação real do MedQuAD na revisão fixa: 1.495 pares únicos elegíveis;
  recorte de 750 (572 treino, 78 validação, 100 teste).
- 30 duplicatas exatas removidas; 225 respostas fora do limite e 1 registro com
  identificador reconhecido excluídos.
- Duas preparações independentes geraram arquivos idênticos byte a byte.
- Sem interseção de documento, grupo, URL, pergunta ou resposta entre splits no recorte.
- Seed executado e reexecutado: 6 pacientes, 5 exames, sem duplicação.
- Exportação interna: 24 exemplos, 8 cenários de avaliação.

O workflow inclui Windows e Linux. Isso é configuração de CI, não afirmação de
execução prévia no computador do usuário. GPU/CUDA, treinamento, qualidade clínica,
tradução do corpus e desempenho do assistente não foram avaliados nesta fase.
