# Correção de contexto RAG — validação v2

Na validação anterior o PR-001 chegou ao contexto, mas o modelo respondeu sobre
fontes (PR-003), sem responder à pergunta sobre identificação. Referências válidas
não significaram uma resposta pertinente.

A indexação agora divide somente o corpo e inclui título e versão em cada trecho,
tanto no embedding quanto no prompt. Não cria vetores separados apenas para o
cabeçalho. Metadados guardam o corpo exato e seu deslocamento no documento original.
A checagem de 128 tokens inclui título, versão e corpo; se exceder, reduza chunk_size.
O prompt pede resposta direta e seleção dos trechos pertinentes, sem alterar pesos.

Índice novo usa schema 2. O leitor mantém suporte ao índice anterior (schema 1),
mas ele não ganha títulos automaticamente. Preserve índice e resultados antigos.

Após atualizar o código, crie uma configuração separada no PowerShell:

```powershell
$c = Get-Content config/rag.json -Raw | ConvertFrom-Json
$c.index = 'data/local/rag-index-v2'
$c.max_distance = 1.15
[System.IO.File]::WriteAllText("$PWD\config\rag-v2.json", ($c | ConvertTo-Json -Depth 10), [System.Text.UTF8Encoding]::new($false))
.\.venv\Scripts\hospital-rag.exe --config config/rag-v2.json index
.\.venv\Scripts\hospital-assess.exe retrieve --config config/rag-v2.json --split validation --output runs/rag-retrieval-v2-01
.\.venv\Scripts\hospital-assess.exe generate --config config/rag-v2.json --split validation --output runs/rag-generation-v2-01
```

O limiar 1.15 é apenas ponto inicial: títulos mudam os vetores e as distâncias.
Reavaliar positivos e pergunta sem cobertura; não calibrar no conjunto de teste.
Comparar results.jsonl das duas versões: resposta à pergunta, fontes efetivamente
usadas, sustentação, recusa fora da base e truncamento. Esta alteração é uma hipótese
de melhoria; somente nova geração permite confirmar seu efeito.

Os testes de chunks verificam título/versão em todos os trechos e posições no
original com LF e CRLF. Não demonstram melhoria semântica do modelo.
