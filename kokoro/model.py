from .istftnet import Decoder
from .modules import CustomAlbert, ProsodyPredictor, TextEncoder
from dataclasses import dataclass
from huggingface_hub import hf_hub_download
from loguru import logger
from numbers import Number
from transformers import AlbertConfig
from typing import Dict, Optional, Union
import json
import torch
import torch.nn as nn


@dataclass
class KModelConfig:
    """Configuration for inner KModel neural network"""

    n_token: int
    hidden_dim: int
    style_dim: int
    n_layer: int
    max_dur: int
    dropout: float
    text_encoder_kernel_size: int
    n_mels: int
    plbert: Dict
    istftnet: Dict
    vocab: Dict[str, int]


class KModelInner(nn.Module):
    REPO_ID = "hexgrad/Kokoro-82M"

    def __init__(self, config: Optional[str] = None, model_path: Optional[str] = None):
        super().__init__()

        # Load config
        if not config:
            logger.debug("No config provided, downloading from HF")
            config = hf_hub_download(repo_id=self.REPO_ID, filename="config.json")

        with open(config, "r", encoding="utf-8") as r:
            config_dict = json.load(r)
            logger.debug(f"Loaded config: {config_dict}")

        self.config = KModelConfig(
            n_token=config_dict["n_token"],
            hidden_dim=config_dict["hidden_dim"],
            style_dim=config_dict["style_dim"],
            n_layer=config_dict["n_layer"],
            max_dur=config_dict["max_dur"],
            dropout=config_dict["dropout"],
            text_encoder_kernel_size=config_dict["text_encoder_kernel_size"],
            n_mels=config_dict["n_mels"],
            plbert=config_dict["plbert"],
            istftnet=config_dict["istftnet"],
            vocab=config_dict["vocab"],
        )

        # Initialize model components
        self.bert = CustomAlbert(
            AlbertConfig(vocab_size=self.config.n_token, **self.config.plbert)
        )
        self.bert_encoder = nn.Linear(
            self.bert.config.hidden_size, self.config.hidden_dim
        )
        self.context_length = self.bert.config.max_position_embeddings
        self.predictor = ProsodyPredictor(
            style_dim=self.config.style_dim,
            d_hid=self.config.hidden_dim,
            nlayers=self.config.n_layer,
            max_dur=self.config.max_dur,
            dropout=self.config.dropout,
        )
        self.text_encoder = TextEncoder(
            channels=self.config.hidden_dim,
            kernel_size=self.config.text_encoder_kernel_size,
            depth=self.config.n_layer,
            n_symbols=self.config.n_token,
        )
        self.decoder = Decoder(
            dim_in=self.config.hidden_dim,
            style_dim=self.config.style_dim,
            dim_out=self.config.n_mels,
            **self.config.istftnet,
        )
        self.load_weights(model_path)

    def load_weights(self, model_path: Optional[str] = None):
        """Load weights from a .pth file or download from HuggingFace"""
        if not model_path:
            logger.debug("No model path provided, downloading from HF")
            model_path = hf_hub_download(
                repo_id=self.REPO_ID, filename="kokoro-v1_0.pth"
            )

        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

        component_map = {
            "bert": self.bert,
            "bert_encoder": self.bert_encoder,
            "predictor": self.predictor,
            "text_encoder": self.text_encoder,
            "decoder": self.decoder,
        }

        for key, inner_dict in state_dict.items():
            if key in component_map:
                try:
                    component_map[key].load_state_dict(inner_dict)
                except:
                    logger.debug(f"Retrying {key} load with modified keys")
                    inner_dict = {k[7:]: v for k, v in inner_dict.items()}
                    component_map[key].load_state_dict(inner_dict, strict=False)
            else:
                logger.debug(f"Unexpected key in state dict: {key}")

    def forward(
        self,
        input_ids: torch.LongTensor,  # [B, T]
        input_lengths: torch.LongTensor,  # [B]
        ref_s: torch.FloatTensor,  # [B, style_dim]
        speed: Number = 1,
    ) -> tuple[torch.FloatTensor, torch.LongTensor, Dict[str, torch.Tensor]]:  # audio, pred_dur, intermediates
        # Create attention mask without device movement
        text_mask = (
            torch.arange(input_lengths.max(), device=input_ids.device)
            .unsqueeze(0)
            .expand(input_lengths.shape[0], -1)
        )
        text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1))

        # BERT encoding
        bert_dur = self.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)

        # Style processing
        s = ref_s[:, 128:]

        # Duration and alignment prediction
        d = self.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = self.predictor.lstm(d)
        duration = self.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()

        # Create alignment matrix
        indices = torch.repeat_interleave(
            torch.arange(input_ids.shape[1], device=input_ids.device), pred_dur
        )
        pred_aln_trg = torch.zeros(
            (input_ids.shape[1], indices.shape[0]), device=input_ids.device
        )
        pred_aln_trg[
            indices, torch.arange(indices.shape[0], device=input_ids.device)
        ] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0)

        # Final generation
        en = d.transpose(-1, -2) @ pred_aln_trg
        F0_pred, N_pred = self.predictor.F0Ntrain(en, s)
        t_en = self.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg
        audio, F0, N, asr_res, har_source, har_spec, har_phase, noi_source, uv, gen_spec, gen_phase, gen_x_0, gen_x_1, gen_x_0_relu, gen_x_0_source_convs, gen_x_0_source_res, gen_x_0_ups = self.decoder(asr, F0_pred, N_pred, ref_s[:, :128])
        audio = audio.squeeze()

        # Collect intermediate values
        intermediates = {
            "bert_dur": bert_dur,
            "d_en": d_en,
            "s": s,
            "d": d,
            "x": x,
            "duration": duration,
            "pred_aln_trg": pred_aln_trg,
            "en": en,
            "F0_pred": F0_pred,
            "N_pred": N_pred,
            "t_en": t_en,
            "asr": asr,
            "F0": F0,
            "N": N,
            "asr_res": asr_res,
            "har_source": har_source,
            "har_spec": har_spec,
            "har_phase": har_phase,
            "noi_source": noi_source,
            "uv": uv,
            "gen_spec": gen_spec,
            "gen_phase": gen_phase,
            "gen_x_0": gen_x_0,
            "gen_x_1": gen_x_1,
            "gen_x_0_relu": gen_x_0_relu,
            "gen_x_0_source_convs": gen_x_0_source_convs,
            "gen_x_0_source_res": gen_x_0_source_res,
            "gen_x_0_ups": gen_x_0_ups,
        }

        return audio, pred_dur, intermediates


class KModel(nn.Module):
    """
    KModel is a torch.nn.Module wrapper around KModelInner that handles:
    1. Converting phonemes to input_ids using the model's vocab
    2. Managing device placement and inference

    You likely only need one KModel instance, and it can be reused across
    multiple KPipelines to avoid redundant memory allocation.

    Unlike KPipeline, KModel is language-blind.
    """

    # Expose REPO_ID from inner model for pipeline compatibility
    REPO_ID = KModelInner.REPO_ID

    def __init__(self, config: Optional[str] = None, model: Optional[str] = None):
        super().__init__()
        self.model = KModelInner(config, model)

    @property
    def vocab(self):
        return self.model.config.vocab

    @property
    def device(self):
        return self.model.bert.device

    @dataclass
    class Output:
        audio: torch.FloatTensor
        pred_dur: Optional[torch.LongTensor] = None

    @torch.no_grad()
    def forward(
        self,
        phonemes: str,
        ref_s: torch.FloatTensor,
        speed: Number = 1,
        return_output: bool = False,
    ) -> Union["KModel.Output", torch.FloatTensor]:
        # Convert phonemes to input ids
        input_ids = list(
            filter(lambda i: i is not None, map(lambda p: self.vocab.get(p), phonemes))
        )
        logger.debug(f"phonemes: {phonemes} -> input_ids: {input_ids}")
        assert len(input_ids) + 2 <= self.model.context_length, (
            len(input_ids) + 2,
            self.model.context_length,
        )

        input_ids = torch.LongTensor([[0, *input_ids, 0]]).to(self.device)
        input_lengths = torch.LongTensor([input_ids.shape[-1]]).to(self.device)
        ref_s = ref_s.to(self.device)

        # Forward through inner model
        audio, pred_dur, intermediates = self.model(input_ids, input_lengths, ref_s, speed)

        audio = audio.cpu()
        pred_dur = pred_dur.cpu() if pred_dur is not None else None

        logger.debug(f"pred_dur: {pred_dur}")

        if return_output:
            return self.Output(audio=audio, pred_dur=pred_dur)
        else:
            return audio
