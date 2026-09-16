"""Integração real opcional: LoRA em Qwen2 minúsculo aleatório, sem rede/model download.

Não é QLoRA nem validação de GPU. A CI específica instala as dependências.
"""
import importlib.util
import tempfile
import unittest
from pathlib import Path

AVAILABLE = all(importlib.util.find_spec(p) for p in ["torch", "transformers", "peft", "accelerate"])


@unittest.skipUnless(AVAILABLE, "Dependências de treino opcionais não instaladas")
class CpuTrainingTests(unittest.TestCase):
    def test_train_save_reload_adapter(self):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM, PreTrainedTokenizerFast, set_seed
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from peft import PeftModel
        from hospital_assistant.finetuning.runtime import add_adapter, fit
        torch.set_num_threads(1)
        set_seed(42)
        architecture = Qwen2Config(vocab_size=16, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=32, bos_token_id=1, eos_token_id=2, pad_token_id=0)
        base = Qwen2ForCausalLM(architecture)
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
            {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3}, unk_token="[UNK]")),
            pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]")
        config = {"seed": 42, "epochs": 1, "learning_rate": 0.001, "gradient_accumulation_steps": 1,
                  "lora_r": 2, "lora_alpha": 4, "lora_dropout": 0.0, "target_modules": ["q_proj", "v_proj"]}
        model = add_adapter(base, config, quantized=False)
        row = {"input_ids": [1, 4, 5, 2], "attention_mask": [1, 1, 1, 1], "labels": [-100, -100, 5, 2]}
        with tempfile.TemporaryDirectory() as tmp:
            result = fit(model, tokenizer, [row, row], [row], config, tmp, steps=2, cpu=True)
            self.assertTrue(result["adapter_weights_changed"])
            self.assertEqual(result["optimizer_steps"], 2)
            self.assertTrue((Path(tmp) / "adapter/adapter_model.safetensors").is_file())
            model.eval()
            inputs = torch.tensor([row["input_ids"]])
            with torch.no_grad():
                expected = model(inputs).logits
            # Recriar base com a mesma seed evita dependência de nomes prefixados pelo PEFT.
            set_seed(42)
            reloaded = PeftModel.from_pretrained(Qwen2ForCausalLM(architecture), Path(tmp) / "adapter")
            reloaded.eval()
            with torch.no_grad():
                actual = reloaded(inputs).logits
            self.assertTrue(torch.allclose(expected, actual, atol=1e-6))
