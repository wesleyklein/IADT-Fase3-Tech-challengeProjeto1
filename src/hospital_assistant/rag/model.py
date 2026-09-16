"""Modelo ajustado local; CPU explícita ou NF4 em CUDA, sem fallback oculto."""
from pathlib import Path
from ..common import read_json


class LocalModel:
    """Carrega tokenizer, pesos e adaptador e encapsula a inferência usada pelo RAG."""
    def __init__(self, config):
        """Inicializa as dependências e a configuração utilizadas pelos métodos desta classe."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        from peft import PeftModel
        run = Path(config['run'])
        metrics = read_json(run / 'metrics.json')
        if metrics.get('status') != 'completed' or metrics.get('mode') != 'train':
            raise ValueError('RAG requer uma execução de treino concluída, não um probe')
        trained = read_json(run / 'resolved_config.json')
        self.tokenizer = AutoTokenizer.from_pretrained(run / 'adapter', trust_remote_code=False)
        if config['device'] == 'cuda':
            from ..finetuning.diagnostics import require_cuda
            from ..finetuning.runtime import load_base
            require_cuda()
            base = load_base(trained)
        else:
            base = AutoModelForCausalLM.from_pretrained(trained['model_id'],
                revision=trained['model_revision'], torch_dtype=torch.float32,
                trust_remote_code=False, use_safetensors=True, attn_implementation='eager')
        self.model = PeftModel.from_pretrained(base, run / 'adapter', is_trainable=False).eval()
        self.config = config
        self.generation = GenerationConfig(do_sample=False, max_new_tokens=config['max_new_tokens'],
            eos_token_id=self.model.generation_config.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            use_cache=True)
        self.identity = {'model_id': trained['model_id'], 'revision': trained['model_revision'],
                         'adapter_run': str(run), 'device': config['device']}

    def encode(self, prompt):
        """Aplica o template de conversa do tokenizer e sinaliza o início da resposta do assistente."""
        messages = [{'role': 'system' if m.type == 'system' else 'user', 'content': m.content}
                    for m in prompt.to_messages()]
        return self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)

    def fits(self, prompt):
        """Verifica se o prompt cabe no limite de entrada, sem cortá-lo silenciosamente."""
        return len(self.encode(prompt)) <= self.config['max_input_tokens']

    def generate(self, prompt):
        """Executa inferência sem gradientes e devolve somente os novos tokens gerados."""
        import torch
        ids = self.encode(prompt)
        if len(ids) > self.config['max_input_tokens']:
            raise ValueError('Prompt excede o orçamento; não foi truncado')
        capacity = self.model.config.max_position_embeddings
        if len(ids) + self.config['max_new_tokens'] > capacity:
            raise ValueError('Prompt e saída excedem o contexto do modelo')
        batch = torch.tensor([ids], device=self.config['device'])
        # Inferência não atualiza pesos; desabilitar gradientes economiza memória.
        with torch.inference_mode():
            generated = self.model.generate(input_ids=batch, attention_mask=torch.ones_like(batch),
                                             generation_config=self.generation)[0, len(ids):]
        eos = self.generation.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        return {'answer': self.tokenizer.decode(generated, skip_special_tokens=True).strip(),
                'input_tokens': len(ids), 'output_tokens': len(generated),
                'truncated': bool(len(generated) and generated[-1].item() not in eos
                                  and len(generated) >= self.config['max_new_tokens'])}

