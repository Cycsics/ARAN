# aran ARAN 🚀, AGPL-3.0 license

from pathlib import Path

from aran.engine.model import Model
from aran.models import aran
from aran.nn.tasks import DetectionModel
from aran.utils import yaml_load, ROOT


class ARAN(Model):
    """ARAN (You Only Look Once) object detection model."""

    def __init__(self, model="ARANv8n.pt", task=None, verbose=False):
        """Initialize ARAN model, switching to ARANWorld if model filename contains '-world'."""
        path = Path(model)
        if "-world" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:  # if ARANWorld PyTorch model
            new_instance = ARANWorld(path)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        else:
            # Continue with default ARAN initialization
            super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "model": DetectionModel,
                "trainer": aran.detect.DetectionTrainer,
                "validator": aran.detect.DetectionValidator,
                "predictor": aran.detect.DetectionPredictor,
            },
        # "segment": {
        #     "model": SegmentationModel,
        #     "trainer": aran.segment.SegmentationTrainer,
        #     "validator": aran.segment.SegmentationValidator,
        #     "predictor": aran.segment.SegmentationPredictor,
        # },
        }

