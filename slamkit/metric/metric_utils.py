
from typing import List
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer

from ..utils.calculation_utils import calc_nll


def get_whisper_pipeline(model_id, device):
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    return pipe


def get_llm(model_id, device):
    tokenzier = AutoTokenizer.from_pretrained(model_id)
    if tokenzier.pad_token_id is None:
        tokenzier.pad_token = tokenzier.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, use_cache=False)
    model.to(device)
    return model, tokenzier


def get_llm_preplexity(model:PreTrainedModel, tokeniser:PreTrainedTokenizer, text:List[str], device) -> torch.FloatTensor:
    model_inputs = tokeniser(text, return_tensors='pt', padding=True).to(device)
    labels = model_inputs['input_ids'].clone()
    labels[labels == tokeniser.pad_token_id] = -100
    with torch.inference_mode():
        logits = model(input_ids=model_inputs['input_ids'], attention_mask=model_inputs['attention_mask']).logits

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return calc_nll(shift_logits, shift_labels, shift_labels.ne(-100))
