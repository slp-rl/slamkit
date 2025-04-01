from typing import List
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer
import re
from ..utils.calculation_utils import calc_nll
from tqdm import tqdm
import os

import logging
logger = logging.getLogger(__name__)


OPENAI_MODELS = [
    "gpt-3.5-turbo",
    "gpt-4",
    "gpt-4o"
]


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

def extract_digit_from_boxed(string):
    match = re.search(r'\\boxed\{(\d+)\}', string)
    if match:
        return int(match.group(1))
    return None

def judge_text(model, tokenizer, text:List[str], device):
    
    tokenizer.padding_side = "left"
    model_inputs = tokenizer(text, return_tensors='pt', padding=True).to(device)
    generation = model.generate(
        input_ids=model_inputs['input_ids'],
        attention_mask=model_inputs['attention_mask'],
        max_new_tokens=512,
        do_sample=True,
        temperature=0.8,
    )
    decode = tokenizer.batch_decode(generation, skip_special_tokens=True)
    res = [extract_digit_from_boxed(text) for text in decode]
    return res

class LLMJudge:
    def __init__(self, model, tokenizer, device, batch_size):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size

    def __call__(self, text:List[str]):
        res = []
        for i in tqdm(range(0, len(text), self.batch_size), desc="LLM Judging"):
            batch_text = text[i:i + self.batch_size]
            res.extend(judge_text(self.model, self.tokenizer, batch_text, self.device))
        return res
    

class OpenAIJudge:
    def __init__(self, name):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
        )
        self.model_name = name
        

    def __call__(self, text:List[str]):
        res = []
        for text in tqdm(text, desc="OpenAI Judging"):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": text}
                    ],
                )
            except Exception as e:
                logger.error(f"Error: {e}")
                continue
            answer = completion.choices[0].message.content
            res.append(extract_digit_from_boxed(answer))
        return res 

        
def get_judge(name, device, batch_size):
    if name in OPENAI_MODELS:
        return OpenAIJudge(name)
    else:
        model, tokenizer = get_llm(name, device)
        return LLMJudge(model, tokenizer, device, batch_size)