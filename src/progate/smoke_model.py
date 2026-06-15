from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any
import importlib
import math
import os
import sys
import traceback

from .io import write_json
from .repro import environment_snapshot, set_seed
from .run_dir import create_run_dir


DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B-Base"
SMOKE_STAGES = ("imports", "load", "forward", "lora", "proposal")
MODEL_CLASSES = ("auto", "causal-lm", "image-text", "auto-image-text")
DTYPES = ("auto", "float32", "float16", "bfloat16")


@dataclass(frozen=True)
class SmokeModelConfig:
    model: str
    seed: int
    output_dir: Path
    project_root: Path
    stage: str
    device: str
    model_class: str
    dtype: str
    device_map: str
    low_cpu_mem_usage: bool
    trust_remote_code: bool
    local_files_only: bool
    max_length: int
    probe_k: int
    max_linear_modules: int


class _StepRecorder(list[dict[str, Any]]):
    def __init__(self, run_dir: Path, config: SmokeModelConfig, package_versions: dict[str, str | None]) -> None:
        super().__init__()
        self._run_dir = run_dir
        self._config = config
        self._package_versions = package_versions

    def append(self, item: dict[str, Any]) -> None:
        super().append(item)
        self.write()

    def write(self, status: str = "running") -> None:
        try:
            write_json(
                self._run_dir / "smoke_progress.json",
                {
                    "status": status,
                    "model": self._config.model,
                    "stage": self._config.stage,
                    "config": _config_summary(self._config),
                    "package_versions": self._package_versions,
                    "steps": list(self),
                },
            )
        except Exception:
            pass


def run_smoke_model(config: SmokeModelConfig) -> Path:
    set_seed(config.seed)
    run_dir = create_run_dir(config.output_dir, "smoke-model", config.seed)
    write_json(run_dir / "env.json", environment_snapshot(config.project_root, sys.argv))

    package_versions = _package_versions(["torch", "transformers", "peft", "accelerate", "datasets", "safetensors"])
    steps = _StepRecorder(run_dir, config, package_versions)
    steps.write()

    try:
        result = _run_checks(config, steps, package_versions)
    except Exception as error:  # noqa: BLE001 - blocker logging must keep the original failure.
        blocker = _blocker(
            blocker_type=_blocker_type(steps, error),
            config=config,
            package_versions=package_versions,
            model_load_class=_last_model_class(steps),
            error=error,
            attempted_fix=_attempted_fix(config),
            next_fix_candidate=_next_fix_candidate(steps),
        )
        write_json(run_dir / "blocker.json", blocker)
        result = {
            "status": "blocked",
            "model": config.model,
            "stage": config.stage,
            "config": _config_summary(config),
            "package_versions": package_versions,
            "gpu_snapshot": _safe_gpu_snapshot(),
            "steps": steps,
            "blocker_path": str(run_dir / "blocker.json"),
        }

    write_json(run_dir / "smoke.json", result)
    steps.write(status=str(result.get("status", "unknown")))
    _write_run_note(run_dir, result)
    return run_dir


def _run_checks(
    config: SmokeModelConfig,
    steps: list[dict[str, Any]],
    package_versions: dict[str, str | None],
) -> dict[str, Any]:
    _validate_stage(config.stage)
    _validate_loader_config(config)
    torch = _import_required("torch", steps)
    transformers = _import_required("transformers", steps)
    peft = _import_required("peft", steps)
    _import_required("accelerate", steps)
    _import_required("datasets", steps)
    _import_required("safetensors", steps)

    if hasattr(torch, "manual_seed"):
        torch.manual_seed(config.seed)

    if config.stage == "imports":
        return _ok_result(config, package_versions, steps, model_class=None, device=None)

    dtype = _resolve_dtype(torch, config.dtype)
    device = _resolve_device(torch, config.device)
    steps.append(
        {
            "name": "resolve_device",
            "status": "ok",
            "device": device,
            "dtype": config.dtype,
            "device_map": _effective_device_map(config),
        }
    )

    tokenizer, processor = _load_text_frontend(transformers, config, steps)
    _load_model_config(transformers, config, steps)
    model, model_class = _load_model(transformers, config, steps, dtype)
    if _uses_device_map(config):
        input_device = _input_device(model, torch, device)
        steps.append({"name": "place_model", "status": "ok", "mode": "device_map", "input_device": input_device})
    else:
        model.to(device)
        input_device = device
        steps.append({"name": "place_model", "status": "ok", "mode": "single_device", "input_device": input_device})
    model.train()

    linear_modules = _linear_module_names(model, torch)
    target_modules = _target_module_names(linear_modules)
    steps.append(
        {
            "name": "inspect_lora_targets",
            "status": "ok",
            "linear_module_count": len(linear_modules),
            "linear_module_sample": linear_modules[: config.max_linear_modules],
            "target_modules": target_modules,
        }
    )
    if not target_modules:
        raise RuntimeError("no LoRA target modules found")

    if config.stage == "load":
        return _ok_result(config, package_versions, steps, model_class=model_class, device=device)

    try:
        batch = _make_text_batch(tokenizer, processor, torch, input_device, config.max_length)
        loss = _forward_loss(model, batch)
    except Exception as error:
        steps.append({"name": "text_forward_loss", "status": "failed", "error": repr(error)})
        raise RuntimeError("failed to run text-only forward loss") from error
    steps.append({"name": "text_forward_loss", "status": "ok", "loss": _float(loss)})

    if config.stage == "forward":
        return _ok_result(config, package_versions, steps, model_class=model_class, device=device)

    try:
        lora_task_type = None if model_class == "AutoModelForImageTextToText" else "CAUSAL_LM"
        lora_exclude_modules = (
            r".*(vision_tower|audio_tower|embed_vision|embed_audio).*"
            if model_class == "AutoModelForImageTextToText"
            else None
        )
        lora_config = peft.LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            bias="none",
            target_modules=target_modules,
            exclude_modules=lora_exclude_modules,
            task_type=lora_task_type,
        )
        model = peft.get_peft_model(model, lora_config)
    except Exception as error:
        steps.append({"name": "apply_lora", "status": "failed", "error": repr(error)})
        raise RuntimeError("failed to attach LoRA adapters") from error
    model.train()
    trainable_named = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    trainable = [param for _, param in trainable_named]
    steps.append(
        {
            "name": "apply_lora",
            "status": "ok",
            "peft_model_class": model.__class__.__name__,
            "lora_task_type": lora_task_type,
            "lora_exclude_modules": lora_exclude_modules,
            "trainable_parameter_count": _parameter_count(trainable),
            "trainable_name_sample": [name for name, _ in trainable_named[:20]],
        }
    )

    try:
        batch = _make_text_batch(tokenizer, processor, torch, input_device, config.max_length)
        loss, forward_diag = _forward_loss_with_diagnostics(model, batch)
        if not forward_diag["loss_requires_grad"]:
            raise RuntimeError(f"LoRA loss is detached: {forward_diag}")
        loss.backward()
        grad_norm = _grad_norm(trainable, torch)
    except Exception as error:
        steps.append({"name": "lora_backward", "status": "failed", "error": repr(error)})
        raise RuntimeError("failed to run LoRA backward pass") from error
    steps.append({"name": "lora_backward", "status": "ok", "loss": _float(loss), "grad_norm": grad_norm, **forward_diag})
    if grad_norm == 0.0:
        raise RuntimeError("LoRA trainable parameters did not receive gradients")

    if config.stage == "lora":
        return _ok_result(config, package_versions, steps, model_class=model_class, device=device)

    try:
        optimizer = torch.optim.AdamW(trainable, lr=1e-4)
        before, updates, q_t_norm = _extract_adamw_proposal(optimizer, trainable, torch)
        optimizer.zero_grad(set_to_none=True)
    except Exception as error:
        steps.append({"name": "adamw_q_t_extraction", "status": "failed", "error": repr(error)})
        raise RuntimeError("failed to extract AdamW proposal") from error
    steps.append({"name": "adamw_q_t_extraction", "status": "ok", "q_t_norm": q_t_norm})
    if q_t_norm == 0.0:
        raise RuntimeError("AdamW step produced zero proposal norm")

    if config.probe_k > 0:
        try:
            probe_stats = _probe_delta(
                model=model,
                tokenizer=tokenizer,
                processor=processor,
                torch=torch,
                params=trainable,
                updates=updates,
                input_device=input_device,
                max_length=config.max_length,
                probe_k=config.probe_k,
            )
        except Exception as error:
            steps.append({"name": "probe_delta", "status": "failed", "error": repr(error)})
            raise RuntimeError("failed to compute K=1 virtual probe delta") from error
        steps.append({"name": "probe_delta", "status": "ok", **probe_stats})

    return _ok_result(config, package_versions, steps, model_class=model_class, device=device)


def _ok_result(
    config: SmokeModelConfig,
    package_versions: dict[str, str | None],
    steps: list[dict[str, Any]],
    model_class: str | None,
    device: str | None,
) -> dict[str, Any]:
    return {
        "status": "ok",
        "model": config.model,
        "stage": config.stage,
        "config": _config_summary(config),
        "model_load_class": model_class,
        "package_versions": package_versions,
        "device": device,
        "gpu_snapshot": _safe_gpu_snapshot(),
        "steps": steps,
    }


def _import_required(module_name: str, steps: list[dict[str, Any]]) -> Any:
    try:
        module = importlib.import_module(module_name)
    except Exception as error:
        steps.append({"name": f"import_{module_name}", "status": "failed"})
        raise RuntimeError(f"failed to import {module_name}") from error
    steps.append({"name": f"import_{module_name}", "status": "ok"})
    return module


def _load_model_config(transformers: Any, config: SmokeModelConfig, steps: list[dict[str, Any]]) -> None:
    try:
        model_config = transformers.AutoConfig.from_pretrained(
            config.model,
            trust_remote_code=config.trust_remote_code,
            local_files_only=config.local_files_only,
        )
    except Exception as error:
        steps.append({"name": "load_config", "status": "failed", "error": repr(error)})
        raise RuntimeError("failed to load model config") from error
    steps.append(
        {
            "name": "load_config",
            "status": "ok",
            "model_type": getattr(model_config, "model_type", None),
            "architectures": getattr(model_config, "architectures", None),
        }
    )


def _load_text_frontend(transformers: Any, config: SmokeModelConfig, steps: list[dict[str, Any]]) -> tuple[Any | None, Any | None]:
    tokenizer = None
    processor = None

    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            config.model,
            trust_remote_code=config.trust_remote_code,
            local_files_only=config.local_files_only,
        )
        steps.append({"name": "load_tokenizer", "status": "ok", "class": tokenizer.__class__.__name__})
    except Exception as error:
        steps.append({"name": "load_tokenizer", "status": "optional_failed", "error": repr(error)})

    try:
        processor = transformers.AutoProcessor.from_pretrained(
            config.model,
            trust_remote_code=config.trust_remote_code,
            local_files_only=config.local_files_only,
        )
        steps.append({"name": "load_processor", "status": "ok", "class": processor.__class__.__name__})
    except Exception as error:
        steps.append({"name": "load_processor", "status": "optional_failed", "error": repr(error)})

    if tokenizer is None and processor is None:
        steps.append({"name": "load_text_frontend", "status": "failed"})
        raise RuntimeError("failed to load tokenizer or processor")
    return tokenizer, processor


def _load_model(transformers: Any, config: SmokeModelConfig, steps: list[dict[str, Any]], dtype: Any | None) -> tuple[Any, str]:
    load_kwargs = _model_load_kwargs(config, dtype)
    steps.append(
        {
            "name": "model_load_options",
            "status": "ok",
            "model_class": config.model_class,
            "dtype": config.dtype,
            "device_map": _effective_device_map(config),
            "low_cpu_mem_usage": config.low_cpu_mem_usage,
            "local_files_only": config.local_files_only,
        }
    )
    available = {
        "AutoModelForCausalLM": transformers.AutoModelForCausalLM,
        "AutoModelForImageTextToText": getattr(transformers, "AutoModelForImageTextToText", None),
    }
    if config.model_class == "auto-image-text":
        order = ["AutoModelForImageTextToText", "AutoModelForCausalLM"]
    elif config.model_class == "image-text":
        order = ["AutoModelForImageTextToText"]
    elif config.model_class == "causal-lm":
        order = ["AutoModelForCausalLM"]
    else:
        order = ["AutoModelForCausalLM", "AutoModelForImageTextToText"]
    loaders = [(class_name, available.get(class_name)) for class_name in order]
    errors: list[dict[str, str]] = []
    for class_name, loader in loaders:
        if loader is None:
            errors.append({"class": class_name, "error": "class unavailable"})
            continue
        try:
            model = loader.from_pretrained(config.model, **load_kwargs)
        except Exception as error:
            errors.append({"class": class_name, "error": repr(error)})
            steps.append({"name": "load_model", "status": "failed", "class": class_name, "error": repr(error)})
            continue
        steps.append({"name": "load_model", "status": "ok", "class": class_name})
        return model, class_name
    raise RuntimeError(f"failed to load model with supported classes: {errors}")


def _model_load_kwargs(config: SmokeModelConfig, dtype: Any | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "trust_remote_code": config.trust_remote_code,
        "local_files_only": config.local_files_only,
    }
    if dtype is not None:
        kwargs["dtype"] = dtype
    device_map = _effective_device_map(config)
    if device_map is not None:
        kwargs["device_map"] = device_map
    if config.low_cpu_mem_usage:
        kwargs["low_cpu_mem_usage"] = True
    return kwargs


def _make_text_batch(
    tokenizer: Any | None,
    processor: Any | None,
    torch: Any,
    device: str,
    max_length: int,
    text: str = "ProGate smoke test.",
) -> dict[str, Any]:
    kwargs = {"return_tensors": "pt", "truncation": True, "max_length": max_length}
    if tokenizer is not None:
        batch = _call_text_frontend(tokenizer, text, kwargs)
    elif processor is not None:
        batch = _call_processor(processor, text, kwargs)
    else:
        raise RuntimeError("no tokenizer or processor available")
    batch = _move_batch(batch, device)
    if "input_ids" not in batch:
        raise RuntimeError("text frontend did not produce input_ids")
    batch["labels"] = batch["input_ids"].clone()
    return batch


def _call_text_frontend(tokenizer: Any, text: str, kwargs: dict[str, Any]) -> Any:
    try:
        return tokenizer(text, **kwargs)
    except TypeError:
        return tokenizer(text, return_tensors="pt")


def _call_processor(processor: Any, text: str, kwargs: dict[str, Any]) -> Any:
    try:
        return processor(text=[text], **kwargs)
    except TypeError:
        return processor(text=[text], return_tensors="pt")


def _move_batch(batch: Any, device: str) -> dict[str, Any]:
    if hasattr(batch, "items"):
        return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
    raise RuntimeError("text frontend did not return a mapping")


def _forward_loss(model: Any, batch: dict[str, Any]) -> Any:
    outputs = model(**batch)
    loss = getattr(outputs, "loss", None)
    if loss is None:
        raise RuntimeError("model output did not include loss")
    return loss


def _forward_loss_with_diagnostics(model: Any, batch: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    outputs = model(**batch)
    loss = getattr(outputs, "loss", None)
    if loss is None:
        raise RuntimeError("model output did not include loss")
    logits = getattr(outputs, "logits", None)
    return loss, {
        "loss_requires_grad": bool(getattr(loss, "requires_grad", False)),
        "loss_grad_fn": loss.grad_fn.__class__.__name__ if getattr(loss, "grad_fn", None) is not None else None,
        "logits_requires_grad": bool(getattr(logits, "requires_grad", False)) if logits is not None else None,
        "logits_grad_fn": logits.grad_fn.__class__.__name__
        if logits is not None and getattr(logits, "grad_fn", None) is not None
        else None,
    }


def _resolve_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_dtype(torch: Any, requested: str) -> Any | None:
    if requested == "auto":
        return None
    if requested == "float32":
        return torch.float32
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unknown dtype: {requested}")


def _effective_device_map(config: SmokeModelConfig) -> str | None:
    if not config.device_map or config.device_map == "none":
        return None
    return config.device_map


def _uses_device_map(config: SmokeModelConfig) -> bool:
    return _effective_device_map(config) is not None


def _input_device(model: Any, torch: Any, fallback: str) -> str:
    if fallback == "cuda" and torch.cuda.is_available():
        return "cuda"
    for param in model.parameters():
        device = getattr(param, "device", None)
        if device is not None and str(device) != "meta":
            return str(device)
    return fallback


def _linear_module_names(model: Any, torch: Any) -> list[str]:
    names: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            names.append(name)
    return names


def _target_module_names(linear_modules: list[str]) -> list[str]:
    preferred = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "in_proj",
        "in_proj_a",
        "in_proj_b",
        "in_proj_qkv",
        "in_proj_z",
        "out_proj",
    }
    direct: set[str] = set()
    wrapped: set[str] = set()
    for name in linear_modules:
        parts = name.split(".")
        leaf = parts[-1]
        if leaf in preferred:
            direct.add(leaf)
        if leaf == "linear" and len(parts) >= 2 and parts[-2] in preferred:
            wrapped.add(f"{parts[-2]}.linear")

    if direct:
        return sorted(direct)
    return sorted(wrapped)


def _parameter_count(params: list[Any]) -> int:
    return sum(param.numel() for param in params)


def _grad_norm(params: list[Any], torch: Any) -> float:
    total = 0.0
    for param in params:
        if param.grad is None:
            continue
        total += float(torch.sum(param.grad.detach() * param.grad.detach()).item())
    return math.sqrt(total)


def _proposal_norm(before: list[Any], params: list[Any], torch: Any) -> float:
    total = 0.0
    for old, param in zip(before, params, strict=True):
        diff = param.detach() - old
        total += float(torch.sum(diff * diff).item())
    return math.sqrt(total)


def _extract_adamw_proposal(optimizer: Any, params: list[Any], torch: Any) -> tuple[list[Any], list[Any], float]:
    before = [param.detach().clone() for param in params]
    optimizer.step()
    updates = [(param.detach() - old).clone() for old, param in zip(before, params, strict=True)]
    q_t_norm = _tensor_list_norm(updates, torch)
    _restore_params(before, params)
    return before, updates, q_t_norm


def _tensor_list_norm(tensors: list[Any], torch: Any) -> float:
    total = 0.0
    for tensor in tensors:
        total += float(torch.sum(tensor * tensor).item())
    return math.sqrt(total)


def _probe_delta(
    model: Any,
    tokenizer: Any | None,
    processor: Any | None,
    torch: Any,
    params: list[Any],
    updates: list[Any],
    input_device: str,
    max_length: int,
    probe_k: int,
) -> dict[str, Any]:
    model.eval()
    deltas: list[float] = []
    base_losses: list[float] = []
    shifted_losses: list[float] = []
    try:
        with torch.no_grad():
            for index in range(probe_k):
                batch = _make_text_batch(
                    tokenizer,
                    processor,
                    torch,
                    input_device,
                    max_length,
                    text=f"ProGate virtual probe {index}.",
                )
                base_loss = _forward_loss(model, batch)
                _apply_updates(params, updates, scale=1.0)
                try:
                    shifted_loss = _forward_loss(model, batch)
                finally:
                    _apply_updates(params, updates, scale=-1.0)
                delta = _float(shifted_loss - base_loss)
                deltas.append(delta)
                base_losses.append(_float(base_loss))
                shifted_losses.append(_float(shifted_loss))
    finally:
        model.train()
    delta_bar = sum(deltas) / len(deltas)
    return {
        "probe_k": probe_k,
        "delta_bar": delta_bar,
        "probe_score": -delta_bar,
        "delta_values": deltas,
        "base_loss_mean": sum(base_losses) / len(base_losses),
        "shifted_loss_mean": sum(shifted_losses) / len(shifted_losses),
    }


def _apply_updates(params: list[Any], updates: list[Any], scale: float) -> None:
    for param, update in zip(params, updates, strict=True):
        param.data.add_(update, alpha=scale)


def _restore_params(before: list[Any], params: list[Any]) -> None:
    for old, param in zip(before, params, strict=True):
        param.data.copy_(old)


def _float(value: Any) -> float:
    return float(value.detach().cpu().item())


def _package_versions(names: list[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _blocker(
    blocker_type: str,
    config: SmokeModelConfig,
    package_versions: dict[str, str | None],
    model_load_class: str | None,
    error: Exception,
    attempted_fix: str,
    next_fix_candidate: str,
) -> dict[str, Any]:
    gpu_snapshot = _safe_gpu_snapshot()
    first_gpu = _first_gpu(gpu_snapshot)
    return {
        "blocker_type": blocker_type,
        "model": config.model,
        "stage": config.stage,
        "config": _config_summary(config),
        "package_versions": package_versions,
        "model_load_class": model_load_class,
        "cuda_visible_devices": gpu_snapshot.get("cuda_visible_devices"),
        "gpu_name": first_gpu.get("name"),
        "gpu_total_memory": first_gpu.get("total_memory_gb"),
        "gpu_peak_memory": first_gpu.get("max_memory_allocated_gb"),
        "gpu_snapshot": gpu_snapshot,
        "error_trace": "".join(traceback.format_exception(error)),
        "attempted_fix": attempted_fix,
        "next_fix_candidate": next_fix_candidate,
    }


def _blocker_type(steps: list[dict[str, Any]], error: Exception) -> str:
    error_text = "".join(traceback.format_exception_only(type(error), error))
    failed = [step for step in steps if step.get("status") == "failed"]
    if not failed:
        message = str(error)
        if _is_oom(error_text):
            return "oom_during_runtime"
        if "no LoRA target modules" in message:
            return "lora_target_not_found"
        if "did not receive gradients" in message:
            return "backward_failed"
        if "zero proposal norm" in message:
            return "optimizer_proposal_failed"
        return "runtime"
    last = failed[-1]
    name = str(last.get("name", "runtime"))
    if name.startswith("import_"):
        return "dependency_missing"
    if name == "load_config":
        return "model_download_failed"
    if name in {"load_tokenizer", "load_processor", "load_text_frontend"}:
        return "processor_or_tokenizer_failed"
    if name == "load_model":
        return _model_load_blocker_type(str(last.get("error", "")))
    if name == "text_forward_loss":
        return "oom_during_forward" if _is_oom(str(last.get("error", ""))) else "text_forward_failed"
    if name == "apply_lora":
        return "peft_attach_failed"
    if name == "lora_backward":
        return "oom_during_backward" if _is_oom(str(last.get("error", ""))) else "backward_failed"
    if name == "adamw_q_t_extraction":
        return "oom_during_backward" if _is_oom(str(last.get("error", ""))) else "optimizer_proposal_failed"
    if name == "probe_delta":
        return "oom_during_forward" if _is_oom(str(last.get("error", ""))) else "probe_delta_failed"
    return "runtime"


def _model_load_blocker_type(error_text: str) -> str:
    lowered = error_text.lower()
    if _is_oom(lowered):
        return "oom_during_load"
    unsupported_markers = [
        "unrecognized configuration",
        "not recognized",
        "not supported",
        "does not support",
        "could not locate the",
        "class unavailable",
    ]
    if any(marker in lowered for marker in unsupported_markers):
        return "model_class_unsupported"
    return "model_download_failed"


def _is_oom(error_text: str) -> bool:
    lowered = error_text.lower()
    markers = [
        "out of memory",
        "cuda oom",
        "cuda error: out of memory",
        "cublas_status_alloc_failed",
        "cuda error: memory allocation",
    ]
    return any(marker in lowered for marker in markers)


def _last_model_class(steps: list[dict[str, Any]]) -> str | None:
    for step in reversed(steps):
        if step.get("name") == "load_model" and "class" in step:
            return str(step["class"])
    return None


def _next_fix_candidate(steps: list[dict[str, Any]]) -> str:
    failed = [step for step in steps if step.get("status") == "failed"]
    if not failed:
        return "inspect traceback and add a targeted smoke test"
    name = str(failed[-1].get("name", "runtime"))
    if name.startswith("import_"):
        return "install or pin the missing ML dependency for Python 3.14.5"
    if name == "load_config":
        return "check model access, cache state, and trust_remote_code"
    if name == "load_model":
        return "try --model-class auto-image-text, --dtype bfloat16, --device-map auto, or local cache loading"
    if name in {"load_tokenizer", "load_processor", "load_text_frontend"}:
        return "inspect tokenizer/processor loading path and try the alternate text frontend"
    if name == "probe_delta":
        return "reduce --max-length or --probe-k, then inspect virtual-probe forward path"
    return "inspect traceback and update the model compatibility path"


def _attempted_fix(config: SmokeModelConfig) -> str:
    return (
        "used configured smoke path with "
        f"model_class={config.model_class}, dtype={config.dtype}, "
        f"device_map={config.device_map}, low_cpu_mem_usage={config.low_cpu_mem_usage}, "
        f"local_files_only={config.local_files_only}"
    )


def _config_summary(config: SmokeModelConfig) -> dict[str, Any]:
    return {
        "seed": config.seed,
        "stage": config.stage,
        "device": config.device,
        "model_class": config.model_class,
        "dtype": config.dtype,
        "device_map": config.device_map,
        "low_cpu_mem_usage": config.low_cpu_mem_usage,
        "trust_remote_code": config.trust_remote_code,
        "local_files_only": config.local_files_only,
        "max_length": config.max_length,
        "probe_k": config.probe_k,
        "max_linear_modules": config.max_linear_modules,
    }


def _safe_gpu_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    try:
        torch = importlib.import_module("torch")
    except Exception as error:  # noqa: BLE001 - best-effort diagnostics only.
        snapshot["error"] = repr(error)
        return snapshot
    try:
        snapshot["cuda_available"] = bool(torch.cuda.is_available())
        snapshot["device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        devices: list[dict[str, Any]] = []
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                devices.append(
                    {
                        "index": index,
                        "name": props.name,
                        "total_memory_gb": props.total_memory / (1024**3),
                        "memory_allocated_gb": torch.cuda.memory_allocated(index) / (1024**3),
                        "memory_reserved_gb": torch.cuda.memory_reserved(index) / (1024**3),
                        "max_memory_allocated_gb": torch.cuda.max_memory_allocated(index) / (1024**3),
                    }
                )
        snapshot["devices"] = devices
    except Exception as error:  # noqa: BLE001 - best-effort diagnostics only.
        snapshot["error"] = repr(error)
    return snapshot


def _first_gpu(snapshot: dict[str, Any]) -> dict[str, Any]:
    devices = snapshot.get("devices")
    if isinstance(devices, list) and devices:
        first = devices[0]
        if isinstance(first, dict):
            return first
    return {}


def _write_run_note(run_dir: Path, result: dict[str, Any]) -> None:
    status = result.get("status", "unknown")
    lines = [
        "# smoke-model",
        "",
        "## Purpose",
        "",
        "Check whether a model can load, run text-only loss, accept LoRA, backpropagate, expose an AdamW proposal, and compute a forward-only probe delta.",
        "",
        "## Model",
        "",
        str(result.get("model")),
        "",
        "## Stage",
        "",
        str(result.get("stage")),
        "",
        "## Status",
        "",
        str(status),
        "",
    ]
    if status == "blocked":
        lines.extend(
            [
                "## Blocker",
                "",
                str(result.get("blocker_path")),
                "",
            ]
        )
    (run_dir / "run_note.md").write_text("\n".join(lines), encoding="utf-8")


def _validate_stage(stage: str) -> None:
    if stage not in SMOKE_STAGES:
        raise ValueError(f"unknown smoke stage: {stage}")


def _validate_loader_config(config: SmokeModelConfig) -> None:
    if config.model_class not in MODEL_CLASSES:
        raise ValueError(f"unknown model class policy: {config.model_class}")
    if config.dtype not in DTYPES:
        raise ValueError(f"unknown dtype: {config.dtype}")
    if config.device_map == "":
        raise ValueError("device_map cannot be empty; use 'none' or 'auto'")
    if config.max_length <= 0:
        raise ValueError("max_length must be positive")
    if config.probe_k < 0:
        raise ValueError("probe_k must be non-negative")
