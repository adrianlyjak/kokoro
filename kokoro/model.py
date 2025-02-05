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


class KModelInner(nn.Module):
    def __init__(self, config: KModelConfig):
        super().__init__()
        self.bert = CustomAlbert(
            AlbertConfig(vocab_size=config.n_token, **config.plbert)
        )
        self.bert_encoder = nn.Linear(self.bert.config.hidden_size, config.hidden_dim)
        self.context_length = self.bert.config.max_position_embeddings
        self.predictor = ProsodyPredictor(
            style_dim=config.style_dim,
            d_hid=config.hidden_dim,
            nlayers=config.n_layer,
            max_dur=config.max_dur,
            dropout=config.dropout,
        )
        self.text_encoder = TextEncoder(
            channels=config.hidden_dim,
            kernel_size=config.text_encoder_kernel_size,
            depth=config.n_layer,
            n_symbols=config.n_token,
        )
        self.decoder = Decoder(
            dim_in=config.hidden_dim,
            style_dim=config.style_dim,
            dim_out=config.n_mels,
            **config.istftnet,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,  # [B, T]
        input_lengths: torch.LongTensor,  # [B]
        ref_s: torch.FloatTensor,  # [B, style_dim]
        speed: Number = 1,
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:  # audio, pred_dur
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
        audio = self.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze()

        return audio, pred_dur


class KModel(nn.Module):
    """
    KModel is a torch.nn.Module with 2 main responsibilities:
    1. Init weights, downloading config.json + model.pth from HF if needed
    2. forward(phonemes: str, ref_s: FloatTensor) -> (audio: FloatTensor)

    You likely only need one KModel instance, and it can be reused across
    multiple KPipelines to avoid redundant memory allocation.

    Unlike KPipeline, KModel is language-blind.

    KModel stores self.vocab and thus knows how to map phonemes -> input_ids,
    so there is no need to repeatedly download config.json outside of KModel.
    """

    REPO_ID = "hexgrad/Kokoro-82M"

    def __init__(
        self, config: Union[Dict, str, None] = None, model: Optional[str] = None
    ):
        super().__init__()
        if not isinstance(config, dict):
            if not config:
                logger.debug("No config provided, downloading from HF")
                config = hf_hub_download(repo_id=KModel.REPO_ID, filename="config.json")
            with open(config, "r", encoding="utf-8") as r:
                config = json.load(r)
                logger.debug(f"Loaded config: {config}")
        self.vocab = config["vocab"]

        # Create inner model
        inner_config = KModelConfig(
            n_token=config["n_token"],
            hidden_dim=config["hidden_dim"],
            style_dim=config["style_dim"],
            n_layer=config["n_layer"],
            max_dur=config["max_dur"],
            dropout=config["dropout"],
            text_encoder_kernel_size=config["text_encoder_kernel_size"],
            n_mels=config["n_mels"],
            plbert=config["plbert"],
            istftnet=config["istftnet"],
        )
        self.model = KModelInner(inner_config)

        # Load weights - exactly as in original
        if not model:
            model = hf_hub_download(repo_id=KModel.REPO_ID, filename="kokoro-v1_0.pth")

        state_dict = torch.load(model, map_location="cpu", weights_only=True)
        for key, inner_dict in state_dict.items():
            # Map to the corresponding inner model component
            if key == "bert":
                try:
                    self.model.bert.load_state_dict(inner_dict)
                except:
                    logger.debug(f"Did not load {key} from state_dict")
                    inner_dict = {k[7:]: v for k, v in inner_dict.items()}
                    self.model.bert.load_state_dict(inner_dict, strict=False)
            elif key == "bert_encoder":
                try:
                    self.model.bert_encoder.load_state_dict(inner_dict)
                except:
                    logger.debug(f"Did not load {key} from state_dict")
                    inner_dict = {k[7:]: v for k, v in inner_dict.items()}
                    self.model.bert_encoder.load_state_dict(inner_dict, strict=False)
            elif key == "predictor":
                try:
                    self.model.predictor.load_state_dict(inner_dict)
                except:
                    logger.debug(f"Did not load {key} from state_dict")
                    inner_dict = {k[7:]: v for k, v in inner_dict.items()}
                    self.model.predictor.load_state_dict(inner_dict, strict=False)
            elif key == "text_encoder":
                try:
                    self.model.text_encoder.load_state_dict(inner_dict)
                except:
                    logger.debug(f"Did not load {key} from state_dict")
                    inner_dict = {k[7:]: v for k, v in inner_dict.items()}
                    self.model.text_encoder.load_state_dict(inner_dict, strict=False)
            elif key == "decoder":
                try:
                    self.model.decoder.load_state_dict(inner_dict)
                except:
                    logger.debug(f"Did not load {key} from state_dict")
                    inner_dict = {k[7:]: v for k, v in inner_dict.items()}
                    self.model.decoder.load_state_dict(inner_dict, strict=False)
            else:
                logger.debug(f"Unexpected key in state dict: {key}")

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
        audio, pred_dur = self.model(input_ids, input_lengths, ref_s, speed)

        audio = audio.cpu()
        pred_dur = pred_dur.cpu() if pred_dur is not None else None

        logger.debug(f"pred_dur: {pred_dur}")

        if return_output:
            return self.Output(audio=audio, pred_dur=pred_dur)
        return audio
