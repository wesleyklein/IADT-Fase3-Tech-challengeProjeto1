# Evidências de validação da fase 2

Ambiente de verificação: Linux, Python 3.12, torch 2.6.0+cpu,
transformers 4.51.3, peft 0.15.2 e accelerate 1.6.0.
Modelo/tokenizer: revisão `7ae557604adf67be50417f59c2c2f167def9a775`.

- **20 testes aprovados**, incluindo os 13 da fase 1.
- Integração real: Qwen2 aleatório minúsculo, LoRA, 2 passos em CPU, mudança dos
  pesos treináveis, loss finita, salvamento safetensors e recarga com logits iguais.
  O propósito é validar o código, não produzir um modelo médico.
- Diagnóstico sem dependências gerou JSON e saída 1 com instruções de instalação.
- Tokenizer real de Qwen2.5-0.5B-Instruct carregado: chat template compatível com
  máscara assistant-only e processamento real dos arquivos da fase 1.

| Limite de tokens | Treino elegível | Validação elegível | Teste elegível |
|---|---:|---:|---:|
| 128 (probe) | 60: 36 EN + 24 PT sintéticos | 4 | 2 |
| 256 (treino padrão) | 259: 235 EN + 24 PT sintéticos | 27 | 43 |

Entrada: 572 pares de treino MedQuAD + 24 internos; 78 validação; 100 teste.
O probe usa no máximo 8 de treino e 2 de validação. O relatório de cada execução
registra sua seleção efetiva. Os exemplos excluídos permanecem nos arquivos
originais; não foram truncados ou apagados.

Código do notebook validado sintaticamente, sem execução no Colab. QLoRA/NF4,
driver/CUDA, pico de memória, treinamento no MedQuAD e comparação base/ajustado
com pesos reais **ainda precisam ser executados na GPU do usuário ou em GPU maior**.
Não há evidência de viabilidade na MX150 nem conclusão clínica nesta PR.
