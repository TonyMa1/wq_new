"""
Alpha Generator package for WorldQuant Brain.

Provides tools for generating, refining, testing, and submitting
alpha expressions to WorldQuant Brain.
"""

from alpha_gen.api.wq_client import WorldQuantClient, WorldQuantError
from alpha_gen.api.ai_client import AIClient, AIClientError
from alpha_gen.models.alpha import Alpha, AlphaMetrics, AlphaCheck, SimulationSettings
from alpha_gen.core.alpha_generator import AlphaGenerator, AlphaGeneratorError
from alpha_gen.core.alpha_polisher import AlphaPolisher, AlphaPolisherError
from alpha_gen.core.alpha_simulator import AlphaSimulator, AlphaSimulatorError
from alpha_gen.core.alpha_submitter import AlphaSubmitter, AlphaSubmitterError
from alpha_gen.utils.config import Config, WorldQuantConfig, AIConfig, AppConfig
from alpha_gen.utils.logging import setup_logging, get_logger

__version__ = '0.1.0'
__author__ = 'TonyMa1'