import torch
from colpali_engine.models import ColPali, ColPaliProcessor

from config import DEVICE, MODEL_NAME

_model = None
_processor = None


def get_model() -> tuple[ColPali, ColPaliProcessor]:
    global _model, _processor
    if _model is None:
        _model = ColPali.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map=DEVICE,
        ).eval()
        _processor = ColPaliProcessor.from_pretrained(MODEL_NAME)
    return _model, _processor
