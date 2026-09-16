# Fase 2 — diagnóstico, QLoRA e comparação

## O que está entregue

Comandos de diagnóstico, teste curto, treinamento supervisionado e avaliação
pareada, com adaptadores, tokenizer, configuração resolvida e métricas salvos.
O teste curto é uma medição de viabilidade: **não conclui o fine-tuning do projeto**.
Não há pesos treinados no MedQuAD nem métricas clínicas inventadas nesta PR.

## 1. Diagnóstico no Windows

Na raiz do repositório, após atualizar a branch:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\hospital-train.exe doctor
```

O comando funciona mesmo sem PyTorch e sem `nvidia-smi`: cria `runs/doctor.json`
com dependências ausentes, disponibilidade CUDA, GPU, VRAM e erros. Saída 1
significa ambiente ainda não pronto, não corrupção dos dados. Envie esse JSON
para diagnosticar. O driver de 2021 mostrado no print pode impedir CUDA 11.8;
atualização de driver deve usar o fabricante/NVIDIA, não é feita pelo script.

## 2. Instalar o ambiente de treinamento

Usar Python **3.11 ou 3.12** para este perfil fixado. Instalar PyTorch CUDA antes
das outras dependências evita obter uma versão somente CPU por engano.

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
.\.venv\Scripts\python.exe -m pip install -r requirements/finetuning.txt
.\.venv\Scripts\hospital-train.exe doctor
```

Esses downloads são grandes. Não é necessário instalar o CUDA Toolkit completo
para usar os wheels precompilados; é necessário um driver compatível. O diagnóstico
executa um kernel PyTorch pequeno, mas somente `probe` verifica também o caminho
NF4/LoRA. Em caso de incompatibilidade de DLL, arquitetura ou driver, mantenha o
relatório e corrija o ambiente antes de prosseguir. Não há fallback silencioso para CPU.

## 3. Teste curto na MX150 (2 GB)

Os arquivos da fase 1 devem existir. Caso necessário: `hospital-data prepare` e
`hospital-data synthetic`. Execute:

```powershell
.\.venv\Scripts\hospital-train.exe probe --output runs/mx150-probe-01
```

O comando baixa Qwen/Qwen2.5-0.5B-Instruct (candidato, não modelo médico), confirma
a revisão SHA configurada e carrega tokenizer e pesos na mesma revisão. Usa NF4 4 bits,
double quantization, LoRA rank 4 em q_proj/v_proj, batch 1, sequência máxima 128,
8 exemplos elegíveis de treino, 2 de validação e **5 passos de otimizador**.

FP32 é usado no cálculo e nas partes não quantizadas para um perfil conservador
sem BF16/FlashAttention. Isso consome memória além dos pesos em 4 bits. Não existe
garantia de caber nos 2 GB. GPU 0 é usada explicitamente; não há offload automático.
Embeddings e lm_head também pesam no orçamento de memória. Se faltar VRAM,
conserve `failure.json` e utilize GPU maior. Não reduza silenciosamente o conteúdo
das respostas nem chame a execução incompleta de treinamento concluído.

Para repetir, escolha outra pasta (por exemplo `runs/mx150-probe-02`). Pastas
existentes não são sobrescritas.

## 4. Treinamento completo

Somente após o probe funcionar e avaliarmos tempo/memória:

```powershell
.\.venv\Scripts\hospital-train.exe train --output runs/qlora-01
```

`config/qlora.json`: 1 época, sequência máxima 256, batch 1, acumulação 8,
learning rate 0.0002, LoRA r=4/alpha=8/dropout=0.05 e gradient checkpointing.
A sequência 256 pode exceder a memória mesmo se o probe em 128 passar. Ajuste
explicitamente uma cópia da configuração e use `--config CAMINHO` se necessário.

O treino reúne originais MedQuAD em inglês e 24 exemplos internos em português.
Isso é um experimento inicial bilíngue; não transforma o MedQuAD em um corpus
português revisado. Para usar traduções aprovadas, altere os três caminhos de
split na configuração; nunca inclua original e tradução do mesmo ID juntos.

Somente tokens da resposta e seu terminador entram na loss; prompt e padding
recebem -100. Exemplos completos maiores que o limite são **excluídos e listados**,
não truncados. O treinamento real pode usar bem menos que 750 exemplos. Confira
`data_report.json`; isso introduz viés para respostas curtas. Se nenhum exemplo
couber, o comando falha. Teste é lido apenas para checar isolamento e hash, não
entra no gradiente ou na avaliação durante o treino. Validação ocorre ao final.

## 5. Comparar modelo base e ajustado

```powershell
.\.venv\Scripts\hospital-train.exe evaluate --run runs/qlora-01 --output runs/comparison-01 --limit 20
```

Usa exemplos de **teste**, mesma ordem e prompts, mesma revisão e quantização.
Carrega uma base com o adaptador e alterna LoRA desligado/ligado para reduzir VRAM.
Geração é determinística (`do_sample=False`); não há chave de API ou juiz externo.
Mede NLL/perplexidade dos tokens das respostas, F1 lexical e tempo; salva respostas
individuais para revisão. F1 e perplexidade não provam qualidade ou segurança
clínica. O relatório informa exclusões e IDs. A seleção favorece respostas curtas.
Não usar teste para escolher hiperparâmetros: use validação para isso.

É possível comparar um probe, mas o relatório identifica `training_mode: probe`.
A demonstração final exige treinamento completo e análise dos resultados.

## Arquivos de execução

| Arquivo | Conteúdo |
|---|---|
| `runs/doctor.json` | Diagnóstico do ambiente |
| `resolved_config.json` | Configuração efetiva e revisão SHA do modelo |
| `hardware.json` | Pacotes, GPU e CUDA do treino |
| `data_report.json` | IDs usados, exclusões, idiomas e hashes dos arquivos |
| `trainer_history.json` | Loss e registros do Trainer |
| `metrics.json` | Passos, tempo, loss, pico de memória e mudança dos pesos LoRA |
| `adapter/` | Adaptador safetensors, configuração PEFT e tokenizer |
| `failure.json` | Erro da execução, se falhar |
| `comparison.json` / `predictions.jsonl` | Comparação agregada e respostas individuais |

O adaptador depende do modelo base; não é um modelo completo independente.
As execuções são ignoradas pelo Git. Preserve a pasta de uma execução concluída
fora do ambiente temporário; no Colab use o download ZIP do notebook. Não há
checkpoints intermediários/resume nesta versão: uma interrupção exige nova execução.

## Colab e testes

`notebooks/phase2_colab.ipynb` executa os mesmos comandos em ambiente isolado.
Recebe ZIP do projeto (download da branch pelo GitHub) e preserva resultados por
download. Não inclui tokens do GitHub nem monta serviços pagos automaticamente.
Escolha manualmente runtime com GPU; disponibilidade depende do ambiente.

Testes offline: `python -m unittest discover -s tests -v`. Com dependências de
treino, inclui LoRA real em modelo Qwen2 minúsculo aleatório, sem download de
pesos: treino, salvamento e recarga do adaptador. Esse teste CPU **não valida
QLoRA/NF4/CUDA**. A CI separada instala as dependências CPU.

## Relação com as aulas e referências

- POSTECH Aula 2: tokenização, quantização 4 bits, PEFT e LoRA. O modelo 0.5B
  substitui o Llama 2 7B da aula para experimentar o limite da MX150.
- [Modelo Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).
- [PEFT e quantização](https://huggingface.co/docs/peft/v0.15.0/developer_guides/quantization).
- [Trainer 4.51.3](https://huggingface.co/docs/transformers/v4.51.3/main_classes/trainer).
- [bitsandbytes 0.45.4: compatibilidade](https://huggingface.co/docs/bitsandbytes/v0.45.4/en/installation).
- [Wheels oficiais PyTorch](https://pytorch.org/get-started/previous-versions/).

Versões fixadas para este experimento, não como recomendação de versões mais
recentes. Dependências transitivas são registradas parcialmente em hardware.json;
para arquivar o ambiente completo, salve `python -m pip freeze` junto à execução.
