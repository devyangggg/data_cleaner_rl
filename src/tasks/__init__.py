from .easy import EasyTask
from .medium import MediumTask
from .hard import HardTask

TASK_REGISTRY = {
    "easy": EasyTask,
    "medium": MediumTask,
    "hard": HardTask,
}
