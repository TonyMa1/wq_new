#!/usr/bin/env python
"""
Script to mine alpha expression variations.

This script provides a command-line interface for generating and testing
parameter variations of successful alpha expressions.
"""

import os
import sys
import argparse
import logging
from typing import List, Dict, Optional
import time
import json

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from alpha_gen.api.wq_client import WorldQuantClient, WorldQuantError
from alpha_gen.core.alpha_generator import AlphaGenerator, AlphaGeneratorError
from alpha_gen.core.alpha_simulator import AlphaSimulator, AlphaSimulatorError
from alpha_gen.models.alpha import Alpha, SimulationResult, SimulationSettings
from alpha_gen.utils.logging import setup_logging, get_logger
from alpha_gen.utils.config import Config
from alpha_gen.utils.validators import validate_alpha_expression

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Mine alpha expression variations')
    
    # Input options
    parser.add_argument('--expression', type=str, required=True,
                       help='Base alpha expression to generate variations from')
    
    # Variation options
    parser.add_argument('--range', type=float, default=0.5,
                       help='Parameter range as percentage (default: 0.5 = ±50%%)')
    parser.add_argument('--max-variations', type=int, default=20,
                       help='Maximum number of variations to generate (default: 20)')
    
    # Testing options
    parser.add_argument('--skip-testing', action='store_true',
                       help='Skip testing of generated variations')
    parser.add_argument('--region', type=str, default='USA',
                       help='Region code (default: USA)')
    parser.add_argument('--universe', type=str, default='TOP3000',
                       help='Universe name (default: TOP3000)')
    parser.add_argument('--max-concurrent', type=int, default=5,
                       help='Maximum concurrent simulations (default: 5)')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='./output',
                       help='Output directory (default: ./output)')
    
    # Logging options
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Logging level (default: INFO)')
    parser.add_argument('--log-file', type=str, default=None,
                       help='Log file (default: auto-generated)')
    
    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    logger = setup_logging(
        log_level=args.log_level,
        log_file=args.log_file,
        log_dir='./logs'
    )
    logger.info("Starting alpha expression mining")
    
    try:
        # Validate base expression
        logger.info(f"Validating base expression: {args.expression}")
        is_valid, error = validate_alpha_expression(args.expression)
        if not is_valid:
            logger.error(f"Invalid base expression: {error}")
            return 1
        
        # Load configuration
        logger.info("Loading configuration")
        config = Config.load()
        
        # Create API client
        logger.info("Creating WorldQuant API client")
        wq_client = WorldQuantClient(
            username=config.wq.username,
            password=config.wq.password,
            max_retries=config.wq.max_retries,
            retry_delay=config.wq.retry_delay,
            timeout=config.wq.timeout
        )
        
        # Create alpha generator
        logger.info("Creating alpha generator")
        generator = AlphaGenerator(
            wq_client=wq_client,
            ai_client=None,  # Not needed for parameter mining
            output_dir=args.output_dir,
            max_concurrent_simulations=args.max_concurrent
        )
        
        # Generate variations
        logger.info(f"Generating variations with range ±{args.range*100}%")
        variations = generator.generate_parameter_variations(
            base_expression=args.expression,
            value_range_percent=args.range,
            max_variations=args.max_variations
        )
        
        logger.info(f"Generated {len(variations)} variations")
        
        # Save raw variations
        timestamp = int(time.time())
        variations_file = os.path.join(args.output_dir, f"variations_{timestamp}.json")
        
        os.makedirs(args.output_dir, exist_ok=True)
        with open(variations_file, 'w') as f:
            json.dump({
                "base_expression": args.expression,
                "variations": variations
            }, f, indent=2)
        
        logger.info(f"Saved variations to {variations_file}")
        
        # Test variations if not skipped
        if not args.skip_testing:
            logger.info("Testing variations")
            
            # Create alpha objects
            alphas = []
            for variation in variations:
                settings = Alpha.SimulationSettings(
                    region=args.region,
                    universe=args.universe
                )
                alpha = Alpha(expression=variation, settings=settings)
                alphas.append(alpha)
            
            # Create simulator
            simulator = AlphaSimulator(
                wq_client=wq_client,
                output_dir=args.output_dir,
                max_concurrent_simulations=args.max_concurrent
            )
            
            # Simulate batch
            results = simulator.simulate_batch(
                alphas=alphas,
                save_results=True,
                filename_prefix=f"variations_{timestamp}"
            )
            
            logger.info(f"Tested {len(results)} variations")
            
            # Extract best variations
            best_variations = []
            for alpha, result in results:
                if "alpha_details" in result and result["alpha_details"]:
                    metrics = result["alpha_details"].get("is", {})
                    sharpe = metrics.get("sharpe", 0)
                    fitness = metrics.get("fitness", 0)
                    turnover = metrics.get("turnover", 0)
                    
                    # Check if metrics meet criteria
                    if abs(sharpe) >= 1.25 and abs(fitness) >= 1.0 and 0.01 <= turnover <= 0.7:
                        best_variations.append({
                            "expression": alpha.expression,
                            "alpha_id": result["alpha_details"].get("id"),
                            "sharpe": sharpe,
                            "fitness": fitness,
                            "turnover": turnover
                        })
            
            # Save best variations
            if best_variations:
                best_file = os.path.join(args.output_dir, f"best_variations_{timestamp}.json")
                with open(best_file, 'w') as f:
                    json.dump(best_variations, f, indent=2)
                
                logger.info(f"Saved {len(best_variations)} best variations to {best_file}")
            else:
                logger.info("No variations met the performance criteria")
        
        logger.info("Alpha mining completed successfully")
        return 0
        
    except WorldQuantError as e:
        logger.error(f"WorldQuant API error: {str(e)}")
        return 1
    except (AlphaGeneratorError, AlphaSimulatorError) as e:
        logger.error(f"Error: {str(e)}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())