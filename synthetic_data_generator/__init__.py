"""Synthetic instruction and DPO dataset generation with quality filters."""

from .labeled import LabeledSample, generate_labeled, split
from .generator import SyntheticDataGenerator, InstructionSample, DPOSample

__version__ = "0.1.0"

__all__ = ['LabeledSample', 'generate_labeled', 'split', 'SyntheticDataGenerator', 'InstructionSample', 'DPOSample']
