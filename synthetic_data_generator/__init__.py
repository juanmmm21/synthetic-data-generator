"""Synthetic instruction and DPO dataset generation with quality filters."""

from .generator import SyntheticDataGenerator, InstructionSample, DPOSample

__version__ = "0.1.0"

__all__ = ['SyntheticDataGenerator', 'InstructionSample', 'DPOSample']
