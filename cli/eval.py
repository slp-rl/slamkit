from slamkit.utils.path_utils import resolve_reference_path
import os
from omegaconf import DictConfig
import hydra
from slamkit.vocoder import vocoder_factory
from slamkit.model import tlm_factory
from slamkit.tokeniser import tokeniser_factory
from slamkit.metric.generative_metric import generate, asr_perplexity, llm_as_judge
from slamkit.metric.modelling_metric import swuggy, salmon, sblimp, storycloze
from slamkit.model import SpeechLM
import torch
import logging
logger = logging.getLogger(__name__)


# disables fast tokenisers parallelism, because it doesn't work with Dataloaders
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


@hydra.main(config_name='eval', config_path='../config', version_base="1.3")
def main(cfg: DictConfig):
    if not cfg.model.pretrained_model:
        logger.warning('No pretrained model specified. please specify a pretrained model with model.pretrained_model=<path>')
    tokeniser = tokeniser_factory(cfg.tokeniser)
    if cfg.model.config_args.vocab_size == -1:
        logger.info('Model vocab_size is -1, thus setting it to tokeniser vocab size')
        cfg.model.config_args.vocab_size = len(tokeniser.text_tokeniser)
    tlm = tlm_factory(cfg.model)
    vocoder = vocoder_factory(cfg.vocoder)
    model = SpeechLM(tlm, tokeniser, vocoder=vocoder, device=cfg.device)

    path = resolve_reference_path(cfg.metric.data_path, cfg.reference_path)
    with torch.inference_mode():
        used_token_modality = cfg.metric.get("used_token_modality", None)
        mean_nll = cfg.metric.get("mean_nll", True)
        cross_modal = cfg.metric.get("cross_modal", False)
        if not cross_modal:
            if cfg.metric.metric_type == 'swuggy':
                res = swuggy(model, path, used_token_modality, mean_nll, cfg.batch_size, cfg.num_workers, cfg.pin_memory, cfg.metric.get("subfolder", False))
            elif cfg.metric.metric_type == 'sblimp':
                res = sblimp(model, path, used_token_modality, mean_nll, cfg.batch_size, cfg.num_workers, cfg.pin_memory, cfg.metric.get("subfolder", False))
            elif cfg.metric.metric_type == 'storycloze':
                res = storycloze(model, path, used_token_modality, mean_nll, cfg.batch_size, cfg.num_workers, cfg.pin_memory, cfg.metric.get("subfolder", False))
            elif cfg.metric.metric_type == 'salmon':
                res = salmon(model, path, used_token_modality, mean_nll, cfg.metric.parts, cfg.batch_size, cfg.num_workers, cfg.pin_memory)
            elif cfg.metric.metric_type == 'generate':
                if cfg.vocoder.vocoder_type is None:
                    logger.warning("You are currently trying to run generation without a vocoder, which will generate tokens, but has no effect. You can use a vocoder by, e.g. setting `vocoder=vocoder_hubert_25`")
                res = generate(model, path, cfg.batch_size, used_token_modality,
                            cfg.metric.prompt_length, cfg.metric.get("min_file_length",None), cfg.metric.get("alignment_folder", None), cfg.metric.get("use_alignment", False),
                            tokeniser.fe_sample_rate, cfg.metric.num_files,
                            cfg.num_workers, cfg.pin_memory, **cfg.metric.get("generate_kwargs", {}))
            elif cfg.metric.metric_type == 'asr_perplexity':
                res = asr_perplexity(model, path, cfg.batch_size, cfg.metric.whisper_model, cfg.metric.llm_name_or_path, used_token_modality,
                                    cfg.metric.prompt_length, cfg.metric.get("min_file_length",None), cfg.metric.get("alignment_folder", None), cfg.metric.get("use_alignment", False),
                                    cfg.metric.auto_bleu_n, tokeniser.fe_sample_rate, cfg.metric.get("num_files", None),
                                    cfg.num_workers, cfg.pin_memory, **cfg.metric.get("generate_kwargs", {}))
            elif cfg.metric.metric_type == 'llm_as_judge':
                res = llm_as_judge(model, path, cfg.batch_size, cfg.metric.whisper_model, cfg.metric.llm_name_or_path, cfg.metric.instruction, used_token_modality,
                                   cfg.metric.prompt_length, cfg.metric.min_file_length, cfg.metric.get("alignment_folder", None), cfg.metric.get("use_alignment", False),
                                   tokeniser.fe_sample_rate, cfg.metric.get("num_files", None),
                                   cfg.num_workers, cfg.pin_memory, **cfg.metric.get("generate_kwargs", {}))
            else:
                raise ValueError(f'Unknown metric type: {cfg.metric.metric_type}')
        else:
            if cfg.metric.metric_type == 'storycloze':
                from slamkit.metric.cross_modal_metric import cm_storycloze
                res = cm_storycloze(model, path, cfg.metric.prompt_modality, cfg.metric.cont_modality, used_token_modality, mean_nll, cfg.batch_size, cfg.num_workers, cfg.pin_memory, cfg.metric.get("subfolder", False))
            elif cfg.metric.metric_type == 'generate':
                from slamkit.metric.cross_modal_generation import generate as cm_generate
                if cfg.vocoder.vocoder_type is None:
                    logger.warning("You are currently trying to run generation without a vocoder, which will generate tokens, but has no effect. You can use a vocoder by, e.g. setting `vocoder=vocoder_hubert_25`")
                res = cm_generate(model, path, cfg.batch_size, used_token_modality, cfg.metric.prompt_modality, cfg.metric.cont_modality,
                            cfg.metric.prompt_length, tokeniser.fe_sample_rate, cfg.metric.num_files,
                            cfg.num_workers, cfg.pin_memory, **cfg.metric.get("generate_kwargs", {}))

    if cfg.metric.metric_type != "generate":
        for key, val in res.items():
            if key == "generate" or key == "prompts":
                continue
            if isinstance(val, list):
                print(f"{key}:")
                for i, v in enumerate(val):
                    print(f"\t{i}: {v}")
            else:
                print(f"{key}: {val}")
    if cfg.metric.get("out_path", False) and "generate" in res and cfg.vocoder.vocoder_type is not None:
        import torchaudio
        os.makedirs(cfg.metric.out_path, exist_ok=True)
        for i, gen in enumerate(res["generate"]):
            if i == cfg.metric.get("num_log", -1):
                print(f"Only saving first {i} samples")
                break
            if isinstance(gen, str):
                out_path = os.path.join(cfg.metric.out_path, f"{cfg.metric.metric_type}_{i}.txt")
                with open(out_path, "w") as f:
                    f.write(gen)
            else:  # Audio output
                if gen.shape[-1] == 0:
                    continue
                out_path = os.path.join(cfg.metric.out_path, f"{cfg.metric.metric_type}_{i}.{cfg.metric.ext}")
                torchaudio.save(out_path, gen.cpu().unsqueeze(0), tokeniser.fe_sample_rate)

    if cfg.logger.report_to == "wandb":
        import wandb
        if cfg.logger.run_id is None:
            raise ValueError('No run_id specified for wandb logging')
        wandb.init(project=cfg.logger.project, entity=cfg.logger.entity, id=cfg.logger.run_id, resume="must")
        if "generate" in res and "prompts" in res and cfg.vocoder is not None:
            logs = {}
            for i, (gen, prompt) in enumerate(zip(res["generate"], res["prompts"])):
                if i == cfg.metric.get("num_log", -1):
                    print(f"Only logging first {i} samples")
                    break
                if gen.shape[-1] == 0:
                    continue
                logs[f'generated/generated_{i}'] = wandb.Audio(gen.squeeze(0).cpu().numpy(),
                                                               caption=f'generated_{i}', sample_rate=tokeniser.fe_sample_rate)
                logs[f'prompt/prompt_{i}'] = wandb.Audio(prompt.squeeze(0).cpu().numpy(),
                                                         caption=f'prompt_{i}', sample_rate=tokeniser.fe_sample_rate)
                if "audio_transcription" in res:
                    logs[f'prompt/prompt_text_{i}'] = res["audio_transcription"][i][0]
                    logs[f'generated/generated_text_{i}'] = res["audio_transcription"][i][1]

            wandb.log(logs)
        for key, val in res.items():
            if key == "generate" or key == "prompts":
                continue
            metric_name = f"{cfg.metric.metric_type}/{os.path.basename(os.path.normpath(cfg.metric.data_path))}"
            wandb.log({f"{metric_name}-{part}": val for part, val in res.items()})


if __name__ == "__main__":
    main()
