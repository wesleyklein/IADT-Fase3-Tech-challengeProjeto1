"""Execução real com Transformers/PEFT. Imports pesados somente sob demanda."""
import math
import time
from functools import partial
from pathlib import Path

from ..common import read_json, write_json
from .data import (assert_disjoint, collate_rows, fingerprints, load_split,
                   tokenize_rows)
from .diagnostics import require_cuda


def validate_config(config):
    """Rejeita hiperparâmetros inválidos antes de carregar os pesos na GPU."""
    for key in ["max_length", "epochs", "learning_rate", "gradient_accumulation_steps", "lora_r", "lora_alpha"]:
        if not isinstance(config[key], (int, float)) or not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"Configuração inválida: {key}")
    for key in ["max_length", "gradient_accumulation_steps", "lora_r", "lora_alpha"]:
        if type(config[key]) is not int:
            raise ValueError(f"{key} deve ser inteiro")
    if not 0 <= config["lora_dropout"] < 1 or not config["target_modules"]:
        raise ValueError("Configuração LoRA inválida")


def load_base(config):
    """Carrega os pesos base em NF4 de 4 bits para reduzir o uso de memória da GPU."""
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    # NF4 comprime os pesos base. Double quant também comprime os fatores de escala.
    # A aritmética continua em float32 para compatibilidade com a GPU utilizada.
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_use_double_quant=True,
                                     bnb_4bit_compute_dtype=torch.float32)
    return AutoModelForCausalLM.from_pretrained(
        config["model_id"], revision=config["model_revision"],
        quantization_config=quantization, device_map={"": 0},
        torch_dtype=torch.float32, attn_implementation="eager",
        trust_remote_code=False, use_safetensors=True)


def add_adapter(model, config, quantized=True):
    """Acopla matrizes LoRA treináveis; os pesos base ficam congelados pelo PEFT."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    model.config.use_cache = False
    if quantized:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
    return get_peft_model(model, LoraConfig(
        r=config["lora_r"], lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"], target_modules=config["target_modules"],
        bias="none", task_type="CAUSAL_LM"))


def tensor_collator(rows, pad_id):
    """Converte o lote com padding em tensores inteiros que o Trainer recebe."""
    import torch
    return {key: torch.tensor(value, dtype=torch.long) for key, value in collate_rows(rows, pad_id).items()}


def fit(model, tokenizer, train, validation, config, output, steps=-1, cpu=False):
    """Único loop de treino usado pelo QLoRA e pelo teste minúsculo em CPU."""
    import torch
    from transformers import Trainer, TrainingArguments
    output = Path(output)
    initial_trainable_weights = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    # Batch 1 reduz memória. Acumulação soma gradientes antes de atualizar o adaptador.
    # Checkpointing recalcula ativações para trocar tempo de processamento por memória.
    args = TrainingArguments(
        output_dir=str(output / "checkpoints"), seed=config["seed"], data_seed=config["seed"],
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config["epochs"], max_steps=steps,
        learning_rate=config["learning_rate"], optim="adamw_torch",
        fp16=False, bf16=False, use_cpu=cpu, dataloader_pin_memory=False,
        dataloader_num_workers=0, gradient_checkpointing=not cpu,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="no", save_strategy="no", logging_steps=1,
        logging_nan_inf_filter=False, report_to=[], remove_unused_columns=False,
        label_names=["labels"], max_grad_norm=1.0)
    trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=validation,
                      data_collator=partial(tensor_collator, pad_id=tokenizer.pad_token_id))
    start = time.perf_counter()
    result = trainer.train()
    elapsed = time.perf_counter() - start
    if not math.isfinite(result.training_loss):
        raise ValueError("Treinamento retornou loss não finita")
    adapter_weights_changed = any(not torch.equal(initial_trainable_weights[n], p.detach().cpu())
                  for n, p in model.named_parameters() if n in initial_trainable_weights)
    if not adapter_weights_changed:
        raise ValueError("Nenhum parâmetro treinável mudou; não registrar treino como concluído")
    metrics = trainer.evaluate()
    if not math.isfinite(metrics["eval_loss"]):
        raise ValueError("Loss de validação não finita")
    model.save_pretrained(output / "adapter", safe_serialization=True)
    tokenizer.save_pretrained(output / "adapter")
    write_json(output / "trainer_history.json", trainer.state.log_history)
    return {"optimizer_steps": trainer.state.global_step, "seconds": elapsed,
            "seconds_per_step": elapsed / max(trainer.state.global_step, 1),
            "training_loss": result.training_loss, "validation": metrics,
            "adapter_weights_changed": adapter_weights_changed}


def train(config_path, output, probe=False):
    """Coordena dados, modelo e treinamento; probe executa apenas uma pequena medição técnica."""
    config = read_json(config_path)
    validate_config(config)
    hardware = require_cuda()
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer, set_seed
    # Fixar SHA antes de carregar tokenizer e pesos; o relatório permite repetir a revisão.
    config["model_revision"] = HfApi().model_info(config["model_id"], revision=config["model_revision"]).sha
    if probe:
        config.update(max_length=min(128, config["max_length"]), gradient_accumulation_steps=1)
    write_json(Path(output) / "resolved_config.json", config)
    write_json(Path(output) / "hardware.json", hardware)
    set_seed(config["seed"])
    train_rows = load_split(config["train_files"], "train")
    validation_rows = load_split(config["validation_files"], "validation")
    test_rows = load_split(config["test_files"], "test")
    assert_disjoint(train_rows, validation_rows, test_rows)
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], revision=config["model_revision"], trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    _, train_data, train_report = tokenize_rows(tokenizer, train_rows, config["max_length"], config["seed"], 8 if probe else None)
    _, validation_data, validation_report = tokenize_rows(tokenizer, validation_rows, config["max_length"], config["seed"], 2 if probe else None)
    write_json(Path(output) / "data_report.json", {"train": train_report, "validation": validation_report,
               "files_sha256": fingerprints(config["train_files"] + config["validation_files"] + config["test_files"])})
    torch.cuda.reset_peak_memory_stats()
    model = add_adapter(load_base(config), config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if any(p.requires_grad and "lora_" not in n for n, p in model.named_parameters()):
        raise ValueError("Há pesos fora de LoRA habilitados para treino")
    metrics = fit(model, tokenizer, train_data, validation_data, config, output, steps=5 if probe else -1)
    metrics.update(status="completed", mode="probe" if probe else "train", quantization="NF4-double-quant",
                   trainable_parameters=trainable, peak_allocated_mib=round(torch.cuda.max_memory_allocated() / 2**20, 2),
                   peak_reserved_mib=round(torch.cuda.max_memory_reserved() / 2**20, 2),
                   note="Métricas técnicas; não constituem validação clínica. Probe não é treinamento final.")
    write_json(Path(output) / "metrics.json", metrics)
    return metrics

