# Ficha dos dados

## Origem

MedQuAD: https://github.com/abachaa/MedQuAD. Revisão fixa em `config/data.json`.
Atribuição: Asma Ben Abacha e Dina Demner-Fushman, BMC Bioinformatics (2019).
O repositório declara CC BY 4.0. As traduções devem preservar atribuição e marcar
que o texto foi alterado; não atribuir traduções aos autores do original.

Recorte NIDDK/NHLBI em inglês, com documentos históricos. Informações médicas e
links podem estar desatualizados. Não foram verificados como orientação clínica
atual. O corpus é material acadêmico de QA, não evidência de competência médica.

## Transformações reproduzíveis

1. Verificar revisão Git e checkout limpo.
2. Ler XML das duas coleções permitidas e preservar texto, URL e identificadores.
3. Normalizar Unicode, entidades HTML e espaços, sem remover negações/unidades.
4. Excluir pares vazios, respostas fora de 40–4.000 caracteres e identificadores
   reconhecidos pelos padrões documentados. Não truncar respostas clínicas.
5. Unir documentos ligados por URL, pergunta ou resposta iguais (casefold), com
   conexões transitivas. Deduplicar pares exatos e fazer split por hash do grupo.
6. Selecionar 750 pares com semente 42. Versionar manifesto, não o corpus bruto.
7. Gerar fila de tradução após o split. Somente aprovações rastreáveis são exportadas.

O agrupamento evita vazamento exato, mas não detecta todas as paráfrases ou fatos
equivalentes. Não equivale a teste de generalização por doença. Os filtros de
tamanho favorecem respostas curtas; o recorte não é representativo de toda medicina.

## Campos

Cada par público possui `id`, `document_id`, `group_id`, `source_url`, `source_file`,
`source_file_sha256`, `collection`, `focus`, `question_type`, `question`, `answer`,
`language`, `synthetic`, `license`, `content_hash`, `split`, `review_status` e
`clinical_review`. A revisão Git está no manifesto. `content_hash` identifica o
par original; a tradução mantém esse vínculo e inclui o texto original.

## Dados internos

Todos os pacientes são artificiais (prefixo SYN), sem CPF, e-mail, endereço ou
nomes de pessoas. Protocolos PR-001 a PR-008 são regras didáticas criadas para o
projeto, não normas reais de hospital. Sem doses, prescrição automática ou
diagnósticos definitivos. Revisão clínica: pendente.

24 exemplos internos de treino cobrem identificação, exames, fontes, aprovação,
laudo, receita, resumo e auditoria. Os 8 cenários de avaliação são perguntas
separadas, com comportamentos esperados, e nunca são exportados ao treino.

## Revisão

Revisão automática não significa revisão humana. Tradução aprovada exige nome de
responsável e data, mas esses campos são autodeclarados no protótipo (sem controle
de autenticação). Revisar significado, negações, números, unidades e completude.
Uma aprovação linguística não altera `clinical_review: pending`.

Regex é uma proteção parcial, não um anonimizador de prontuários. Dados reais
estão fora de escopo desta versão.
