#!/usr/bin/env python
"""
Script to generate alpha expressions using AI and test them in WorldQuant Brain.

This script provides a command-line interface for generating alphas with various
parameters and saving the results.
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
from alpha_gen.api.ai_client import AIClient, AIClientError
from alpha_gen.core.alpha_generator import AlphaGenerator, AlphaGeneratorError
from alpha_gen.models.alpha import Alpha
from alpha_gen.utils.logging import setup_logging, get_logger
from alpha_gen.utils.config import Config

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Generate alpha expressions and test them')
    
    # Basic options
    parser.add_argument('--region', type=str, default='USA',
                       help='Region code (default: USA)')
    parser.add_argument('--universe', type=str, default='TOP3000',
                       help='Universe name (default: TOP3000)')
    parser.add_argument('--count', type=int, default=5,
                       help='Number of expressions to generate (default: 5)')
    parser.add_argument('--output-dir', type=str, default='./output',
                       help='Output directory (default: ./output)')
    
    # Advanced generation options
    parser.add_argument('--strategy-type', type=str, default=None,
                       help='Strategy type to focus on (e.g., momentum, value, etc.)')
    parser.add_argument('--data-field-focus', type=str, default=None,
                       help='Comma-separated list of data fields to focus on')
    parser.add_argument('--complexity', type=str, default=None, choices=['simple', 'moderate', 'complex'],
                       help='Complexity level (simple, moderate, complex)')
    
    # Testing options
    parser.add_argument('--skip-testing', action='store_true',
                       help='Skip testing of generated expressions')
    parser.add_argument('--save-variations', action='store_true',
                       help='Save parameter variations of generated expressions')
    parser.add_argument('--max-concurrent', type=int, default=5,
                       help='Maximum concurrent simulations (default: 5)')
    
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
    logger.info("Starting alpha expression generation")
    
    try:
        # Load configuration
        logger.info("Loading configuration")
        config = Config.load()
        
        # Create API clients
        logger.info("Creating API clients")
        wq_client = WorldQuantClient(
            username=config.wq.username,
            password=config.wq.password,
            max_retries=config.wq.max_retries,
            retry_delay=config.wq.retry_delay,
            timeout=config.wq.timeout
        )
        
        ai_client = AIClient(
            api_key=config.ai.api_key,
            model=config.ai.model,
            max_retries=config.ai.max_retries,
            timeout=config.ai.timeout,
            site_url=config.ai.site_url,
            site_name=config.ai.site_name
        )
        
        # Create alpha generator
        logger.info("Creating alpha generator")
        generator = AlphaGenerator(
            wq_client=wq_client,
            ai_client=ai_client,
            output_dir=args.output_dir,
            max_concurrent_simulations=args.max_concurrent
        )
        
        # Parse data field focus if provided
        data_field_focus = None
        if args.data_field_focus:
            data_field_focus = [field.strip() for field in args.data_field_focus.split(',')]
        
        # Generate expressions
        logger.info(f"Generating {args.count} alpha expressions")
        alphas = generator.generate_expressions(
            region=args.region,
            universe=args.universe,
            strategy_type=args.strategy_type,
            data_field_focus=data_field_focus,
            complexity=args.complexity,
            count=args.count
        )
        
        logger.info(f"Generated {len(alphas)} alpha expressions")
        
        # Save raw expressions
        timestamp = int(time.time())
        expressions_file = os.path.join(args.output_dir, f"generated_expressions_{timestamp}.json")
        
        with open(expressions_file, 'w') as f:
            expressions = [{"expression": alpha.expression} for alpha in alphas]
            json.dump(expressions, f, indent=2)
        
        logger.info(f"Saved expressions to {expressions_file}")
        
        # Test expressions if not skipped
        if not args.skip_testing:
            logger.info("Testing generated expressions")
            results = generator.test_expressions(
                alphas=alphas,
                save_results=True
            )
            
            logger.info(f"Tested {len(results)} expressions")
            
            # Generate parameter variations if requested
            if args.save_variations and results:
                logger.info("Generating parameter variations")
                
                variations_file = os.path.join(args.output_dir, f"parameter_variations_{timestamp}.json")
                all_variations = {}
                
                for alpha, result in results:
                    # Only generate variations for alphas with promising metrics
                    if ("alpha_details" in result and 
                        result["alpha_details"] and 
                        "is" in result["alpha_details"]):
                        
                        metrics = result["alpha_details"]["is"]
                        if (abs(metrics.get("sharpe", 0)) >= 0.5 and 
                            metrics.get("turnover", 0) >= 0.01):
                            
                            variations = generator.generate_parameter_variations(
                                base_expression=alpha.expression,
                                value_range_percent=0.5,
                                max_variations=10
                            )
                            
                            all_variations[alpha.id or f"expr_{len(all_variations)}"] = {
                                "original": alpha.expression,
                                "variations": variations
                            }
                
                with open(variations_file, 'w') as f:
                    json.dump(all_variations, f, indent=2)
                
                logger.info(f"Saved parameter variations to {variations_file}")
        
        logger.info("Alpha generation completed successfully")
        return 0
        
    except (WorldQuantError, AIClientError, AlphaGeneratorError) as e:
        logger.error(f"Error: {str(e)}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())