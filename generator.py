"""Compatibilidad: el código vive en el paquete `synthetic_data_generator`. Este módulo se
conserva para quien añade esta carpeta a `sys.path` e importa `generator`
directamente."""

from synthetic_data_generator.generator import *  # noqa: F401,F403
from synthetic_data_generator.generator import SyntheticDataGenerator, InstructionSample, DPOSample  # noqa: F401
