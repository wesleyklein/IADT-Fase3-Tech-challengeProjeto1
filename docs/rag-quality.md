# Seleção de modelo/prompt e respostas sem conteúdo

A comparação mostrou que o adaptador com prompt curto produziu apenas `[PR-005]`.
A referência era válida, mas não havia resposta. Agora `content_missing` detecta
saídas vazias ou contendo somente referências/pontuação. O status será
`content_review_required`; a tela mostra um aviso e mantém a revisão humana.
`content_missing_rate` em generate/compare usa somente gerações concluídas, separado
da taxa de problemas de citação. Esta checagem não verifica correção semântica:
texto incorreto ou repetição do prompt ainda exige leitura humana.

Na tela Streamlit, a barra lateral permite escolher o arquivo de configuração,
modelo Base/Com adaptador e prompt Curto/Atual. Base + curto é a seleção inicial
para avaliação, não uma declaração de qualidade geral. Para usar o índice v2 já
criado, informe `config/rag-v2.json`. Alterações valem para novas solicitações;
rascunhos e decisões já salvos não são regenerados. A troca libera a referência ao
serviço anterior e o próximo uso carrega o modelo novamente.

CLI e avaliação também respeitam os campos opcionais `model_variant` (base/adapted)
e `prompt_variant` (current/short) no JSON. Configurações antigas mantêm adapted/current.
Base significa o mesmo modelo carregado, com adaptador temporariamente desativado;
o diretório de treinamento concluído ainda é necessário. Compare continua executando
as quatro combinações, independentemente da seleção no JSON.

Após o merge, valide:

```powershell
git pull
.\.venv\Scripts\python.exe -m unittest discover -s tests_phase5 -v
$c = Get-Content config/rag-v2.json -Raw | ConvertFrom-Json
$c | Add-Member -NotePropertyName model_variant -NotePropertyValue base -Force
$c | Add-Member -NotePropertyName prompt_variant -NotePropertyValue short -Force
[System.IO.File]::WriteAllText("$PWD\config\rag-candidato.json", ($c | ConvertTo-Json -Depth 10), [System.Text.UTF8Encoding]::new($false))
.\.venv\Scripts\hospital-assess.exe generate --config config/rag-candidato.json --split test --output runs/rag-test-base-short-01
```

O comando final executa a avaliação separada de teste, depois da escolha na validação.
Preserve os arquivos e registre falhas sem ajustar repetidamente a configuração pelo
resultado de teste. Para repetir, use outra pasta. Leia results.jsonl e faça a revisão
humana; métricas automáticas não demonstram adequação clínica nem melhoria do treino.
Os testes automatizados usam modelo simulado e não medem a qualidade real do Qwen.
