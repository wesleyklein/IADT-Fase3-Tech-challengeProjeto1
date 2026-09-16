# Assistente hospitalar acadêmico

Protótipo acadêmico que integra:

- consulta de pacientes e exames fictícios em SQLite;
- busca de protocolos com RAG e LangChain;
- modelo local `Qwen2.5-0.5B-Instruct` com adaptador QLoRA;
- fluxo e revisão humana com LangGraph.

Todo conteúdo é sintético, didático e sem validade clínica.

## 1. Instalação rápida

Requisitos: **Windows, PowerShell, Git, Python 3.11 e internet**.

```powershell
git clone https://github.com/wesleyklein/-IADT-Fase3-Tech-challengeProjeto1.git
cd ./-IADT-Fase3-Tech-challengeProjeto1
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements/workflow.txt
.\.venv\Scripts\hospital-data.exe seed
```

O último comando cria o banco local com **6 pacientes fictícios e 5 exames**.
Se o projeto já estiver clonado, execute `git pull` e repita a instalação das
dependências somente se necessário.

## 2. Preparar a consulta com IA

A pasta abaixo deve existir para usar as opções **Base** e **Com adaptador**:

```text
runs/qlora-mx150-01/
├── metrics.json
├── resolved_config.json
└── adapter/
```

Em `metrics.json`, o treinamento deve estar com `status: completed` e
`mode: train`.

Crie o índice dos protocolos:

```powershell
.\.venv\Scripts\hospital-rag.exe --config config/rag-v2.json index
```

A primeira execução pode demorar porque baixa o modelo de embeddings.

> Se os artefatos do adaptador não foram entregues, consulte a seção
> [Treinar novamente o adaptador](#treinar-novamente-o-adaptador).

## 3. Abrir a aplicação

```powershell
.\.venv\Scripts\python.exe -m streamlit run src/hospital_assistant/workflow/ui.py --server.fileWatcherType none
```

Se aparecer `Email:`, deixe vazio e pressione Enter. Depois, abra
[http://localhost:8501](http://localhost:8501) e mantenha o terminal em execução.

Na lateral da tela, use inicialmente:

- **Arquivo:** `config/rag-v2.json`
- **Modelo:** `Com adaptador`
- **Prompt:** `Curto`

### O que significa “Com adaptador”?

O Qwen é o modelo-base. O adaptador contém os pequenos conjuntos de pesos
aprendidos no fine-tuning QLoRA. Selecionar **Com adaptador** aplica essa
especialização ao modelo-base. No JSON, confirme:

```json
"adapter_enabled": true
```

## 4. Passo a passo para validar pela tela

### Teste 1 — Consulta estruturada de exames

Preencha:

| Campo | Valor |
|---|---|
| O que deseja fazer? | `Consultar exames` |
| Identificador do paciente | `SYN-P001` |
| Pergunta | Pode ficar vazia |

Clique em **Iniciar solicitação**.

Resultado esperado: paciente fictício 001, exame **Hemograma**, status `pending`
e resultado não registrado. Essa operação consulta diretamente o SQLite; a
pergunta não é enviada à LLM.

### Teste 2 — LLM contextualizada com paciente e protocolo

Preencha:

| Campo | Valor |
|---|---|
| O que deseja fazer? | `Consultar protocolo` |
| Identificador do paciente | `SYN-P002` |
| Pergunta | `Segundo o protocolo, o que significa o estado atual do exame?` |

Clique em **Iniciar solicitação** e aguarde. Em CPU, a primeira geração pode
demorar alguns minutos.

Resultado esperado:

> O exame Glicemia está com status completed, que indica resultado registrado
> conforme [PR-002]. Há resultado registrado, sem interpretação clínica.

No JSON, confira:

- `patient_context.patient.id`: `SYN-P002`;
- `patient_context.exams[0].name`: `Glicemia`;
- `patient_context.exams[0].status`: `completed`;
- `patient_source.tool`: `patient_lookup`;
- `model.adapter_enabled`: `true`;
- `generation_performed`: `true`;
- protocolo recuperado: `PR-002`.

Esse teste demonstra o pipeline principal:

```text
pergunta + paciente
→ consulta ao SQLite
→ recuperação de protocolos pelo RAG
→ LLM contextualizada
→ validação do grounding
→ revisão humana
```

Se a LLM produzir um estado diferente do banco, a aplicação mantém a geração em
`original_answer` para auditoria e apresenta uma resposta reconstruída das fontes.

### Teste 3 — Paciente com exame pendente

| Campo | Valor |
|---|---|
| O que deseja fazer? | `Consultar protocolo` |
| Identificador do paciente | `SYN-P001` |
| Pergunta | `Quais exames esse paciente possui e qual é a situação deles?` |

Resultado esperado: **Hemograma**, status `pending`, sem resultado registrado.
Não devem aparecer exames que não estejam em `patient_context`.

### Teste 4 — Paciente sem exames

| Campo | Valor |
|---|---|
| O que deseja fazer? | `Consultar protocolo` |
| Identificador do paciente | `SYN-P003` |
| Pergunta | `Este paciente possui exames pendentes?` |

Resultado esperado: o sistema deve informar que não há exames registrados. Não
deve inventar exames ou diagnósticos.

### Teste 5 — Validações de segurança

| Teste | Preenchimento | Resultado esperado |
|---|---|---|
| Paciente inexistente | `Consultar paciente` + `SYN-P999` | Solicitação bloqueada |
| Rascunho de laudo | `SYN-P001` | Resultado ausente e nenhum diagnóstico inventado |
| Formulário de receita | `SYN-P001` | Apenas campos para preenchimento profissional |

## 5. Revisão humana e persistência

Toda solicitação válida chega a `awaiting_review`.

1. Leia o texto e abra o JSON das fontes.
2. Informe o nome do revisor.
3. Selecione **Aprovar rascunho didático** ou **Rejeitar**.
4. Registre a decisão.

Para validar a persistência:

1. Copie o UUID mostrado na linha **ID**.
2. Cole em **Retomar solicitação → ID da solicitação**.
3. Clique em **Abrir**.
4. Confira se o rascunho e a decisão continuam salvos.

Use o UUID da solicitação, e não códigos como `SYN-P001` ou `PR-002`. Uma
solicitação rejeitada não pode ser aprovada posteriormente.

## 6. Outros testes úteis

| Paciente | Massa disponível |
|---|---|
| `SYN-P001` | Hemograma `pending` |
| `SYN-P002` | Glicemia `completed` |
| `SYN-P003` | Nenhum exame |
| `SYN-P004` | Exame de urina `cancelled` |
| `SYN-P005` | Hemograma `completed` e Glicemia `pending` |

Perguntas sugeridas:

- `Este paciente possui exames pendentes?`
- `Faça um resumo dos exames do paciente seguindo o protocolo de resumo.`
- `Há alguma pendência que precisa ser destacada?`
- `O que é possível afirmar com base nos dados disponíveis?`

As respostas devem usar somente o paciente selecionado, os exames do
`patient_context` e os protocolos apresentados em `context_sources`.

## 7. Testes automatizados

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m unittest discover -s tests_phase5 -v
```

O resultado esperado é **OK** nas duas suítes.

Para executar a avaliação do RAG:

```powershell
.\.venv\Scripts\hospital-assess.exe generate --config config/rag-v2.json --split test --output runs/rag-test-01
```

Use outra pasta de saída ao repetir. O comando não sobrescreve uma execução
existente.

## 8. Treinar novamente o adaptador

Esta etapa exige GPU NVIDIA com CUDA e memória suficiente:

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
.\.venv\Scripts\hospital-data.exe download
.\.venv\Scripts\hospital-data.exe prepare
.\.venv\Scripts\hospital-data.exe synthetic
.\.venv\Scripts\hospital-train.exe doctor
.\.venv\Scripts\hospital-train.exe train --config config/qlora-mx150.json --output runs/qlora-mx150-01
```

Prossiga somente se o diagnóstico confirmar CUDA. O download é necessário apenas
quando `data/raw/MedQuAD` ainda não existir.

## 9. Problemas comuns

- **Modelo ou adaptador ausente:** confira `runs/qlora-mx150-01/`.
- **Índice ausente:** execute o comando `hospital-rag ... index` da seção 2.
- **Execução lenta:** o modelo está em CPU; o primeiro carregamento é mais demorado.
- **Erro de UUID:** use o ID mostrado pela solicitação.
- **Arquivo `.lock` bloqueado:** aguarde a geração atual. Se persistir, encerre todas
  as instâncias com `Ctrl+C`, confirme que não há outro processo executando e
  reinicie a aplicação.
- **Tela mostra `awaiting_review`:** o processamento terminou e aguarda a decisão
  humana; role a página para baixo.
- O parâmetro `--server.fileWatcherType none` evita conflitos do monitor do
  Streamlit com arquivos do PyTorch.

## Dados e referências

- Pacientes e exames: `data/synthetic/hospital.json`
- Protocolos: `data/synthetic/protocols/`
- Perguntas de avaliação: `data/evaluation/rag_cases.json`
- Consulta direta: `.\.venv\Scripts\hospital-data.exe patient SYN-P001`

O treinamento utiliza o [MedQuAD](https://github.com/abachaa/MedQuAD), de Asma
Ben Abacha e Dina Demner-Fushman, sob licença CC BY 4.0. Consulte também a
[ficha de dados](docs/data-card.md). A tradução e a validação clínica permanecem
pendentes.
