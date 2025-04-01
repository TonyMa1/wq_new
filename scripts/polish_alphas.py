#!/usr/bin/env python
"""
Script to polish existing alpha expressions using AI.

This script provides a command-line interface for refining alpha expressions
with specific requirements and testing the improvements.
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
from alpha_gen.core.alpha_polisher import AlphaPolisher, AlphaPolisherError
from alpha_gen.models.alpha import Alpha, SimulationSettings
from alpha_gen.utils.logging import setup_logging, get_logger
from alpha_gen.utils.config import Config

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Polish alpha expressions with AI')
    
    # Input options
    parser.add_argument('--input', type=str, required=True,
                       help='Input file with expressions to polish or single expression')
    parser.add_argument('--input-format', type=str, default='auto', choices=['file', 'json', 'expression', 'auto'],
                       help='Input format (default: auto-detect)')
    
    # Polishing options
    parser.add_argument('--requirements', type=str, default=None,
                       help='Specific requirements for polishing (e.g., "Reduce turnover, improve IR")')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze expressions before polishing')
    
    # Region/universe settings
    parser.add_argument('--region', type=str, default='USA',
                       help='Region code (default: USA)')
    parser.add_argument('--universe', type=str, default='TOP3000',
                       help='Universe name (default: TOP3000)')
    
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

def load_expressions(input_path, input_format='auto'):
    """Load expressions from file or single expression."""
    if input_format == 'auto':
        # Auto-detect format
        if os.path.isfile(input_path):
            if input_path.endswith('.json'):
                input_format = 'json'
            else:
                input_format = 'file'
        else:
            input_format = 'expression'
    
    expressions = []
    
    if input_format == 'expression':
        # Single expression
        expressions = [input_path]
    elif input_format == 'file':
        # Text file with one expression per line
        with open(input_path, 'r') as f:
            expressions = [line.strip() for line in f if line.strip()]
    elif input_format == 'json':
        # JSON file with expressions
        with open(input_path, 'r') as f:
            data = json.load(f)
            
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        expressions.append(item)
                    elif isinstance(item, dict) and 'expression' in item:
                        expressions.append(item['expression'])
            elif isinstance(data, dict) and 'expressions' in data:
                expressions = data['expressions']
    
    return expressions

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
    logger.info("Starting alpha expression polishing")
    
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
        
        # Create alpha polisher
        logger.info("Creating alpha polisher")
        polisher = AlphaPolisher(
            wq_client=wq_client,
            ai_client=ai_client
        )
        
        # Load expressions
        logger.info(f"Loading expressions from {args.input}")
        expressions = load_expressions(args.input, args.input_format)
        
        if not expressions:
            logger.error("No expressions found")
            return 1
        
        logger.info(f"Loaded {len(expressions)} expressions")
        
        # Create Alpha objects
        alphas = []
        for expr in expressions:
            settings = SimulationSettings(
                region=args.region,
                universe=args.universe
            )
            alpha = Alpha(expression=expr, settings=settings)
            alphas.append(alpha)
        
        # Prepare results dictionary
        results = {}
        
        # Process each alpha
        for i, alpha in enumerate(alphas):
            logger.info(f"Processing expression {i+1}/{len(alphas)}")
            
            # Analyze if requested
            if args.analyze:
                logger.info("Analyzing expression")
                try:
                    analysis = polisher.analyze_alpha(alpha, include_metrics=True)
                    
                    results[f"alpha_{i+1}"] = {
                        "original_expression": alpha.expression,
                        "analysis": analysis
                    }
                    
                    logger.info("Analysis complete")
                except AlphaPolisherError as e:
                    logger.error(f"Analysis failed: {str(e)}")
                    results[f"alpha_{i+1}"] = {
                        "original_expression": alpha.expression,
                        "analysis_error": str(e)
                    }
                    continue
            
            # Polish alpha
            logger.info("Polishing expression")
            try:
                polished_alpha, comparison = polisher.polish_alpha(alpha, args.requirements)
                
                # Update results
                if f"alpha_{i+1}" not in results:
                    results[f"alpha_{i+1}"] = {
                        "original_expression": alpha.expression
                    }
                
                results[f"alpha_{i+1}"]["polished_expression"] = polished_alpha.expression
                results[f"alpha_{i+1}"]["comparison"] = comparison
                
                logger.info("Polishing complete")
            except AlphaPolisherError as e:
                logger.error(f"Polishing failed: {str(e)}")
                if f"alpha_{i+1}" not in results:
                    results[f"alpha_{i+1}"] = {
                        "original_expression": alpha.expression
                    }
                results[f"alpha_{i+1}"]["polishing_error"] = str(e)
        
        # Save results
        timestamp = int(time.time())
        results_file = os.path.join(args.output_dir, f"polishing_results_{timestamp}.json")
        
        os.makedirs(args.output_dir, exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved results to {results_file}")
        logger.info("Alpha polishing completed successfully")
        return 0
        
    except (WorldQuantError, AIClientError, AlphaPolisherError) as e:
        logger.error(f"Error: {str(e)}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())