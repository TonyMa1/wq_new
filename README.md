# WorldQuant Alpha Generator

A comprehensive toolkit for generating, refining, testing, and submitting alpha expressions to WorldQuant Brain.

## Overview

This project provides a refactored, modular approach to alpha expression management, emphasizing robust error handling, code organization, and ease of use. It replaces the previous codebase which had significant duplication, inconsistent APIs, and mixed responsibilities.

The new architecture offers:

- Unified API clients for WorldQuant Brain and AI services
- Proper environment-based configuration
- Comprehensive error handling and logging
- Clean separation of concerns

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/tonyma1/worldquant-alpha-generator.git
   cd worldquant-alpha-generator
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the package:
   ```bash
   pip install -e .
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

## Configuration

All configuration is managed through environment variables. Copy `.env.example` to `.env` and update the values:

```env
# WorldQuant Brain API credentials
WQ_USERNAME=your_username
WQ_PASSWORD=your_password

# OpenRouter AI API settings
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=google/gemini-2.5-pro-exp-03-25:free
```

## Usage

### Command Line Scripts

#### Generate Alpha Expressions
```bash
python -m scripts.generate_alphas --region USA --universe TOP3000 --count 5 --strategy-type momentum
```

#### Polish Existing Alphas
```bash
python -m scripts.polish_alphas --input expressions.json --requirements "Reduce turnover, improve IR"
```

#### Mine Expression Variations
```bash
python -m scripts.mine_expressions --expression "rank(ts_mean(close, 10) / ts_mean(close, 20))" --range 0.5
```

#### Submit Successful Alphas
```bash
python -m scripts.submit_alphas --mode auto --sharpe-threshold 1.5 --tags "automated,momentum"
```

### API Usage

```python
from alpha_gen.api.wq_client import WorldQuantClient
from alpha_gen.api.ai_client import AIClient
from alpha_gen.core.alpha_generator import AlphaGenerator
from alpha_gen.utils.config import Config

# Load configuration from environment variables
config = Config.load()

# Create API clients
wq_client = WorldQuantClient(
    username=config.wq.username,
    password=config.wq.password
)

ai_client = AIClient(
    api_key=config.ai.api_key,
    model=config.ai.model
)

# Create alpha generator
generator = AlphaGenerator(
    wq_client=wq_client,
    ai_client=ai_client
)

# Generate expressions
alphas = generator.generate_expressions(
    region='USA',
    universe='TOP3000',
    strategy_type='momentum',
    count=5
)

# Test expressions
results = generator.test_expressions(alphas)
```

## Project Structure

```
worldquant-alpha-generator/
├── alpha_gen/  # Core package
│   ├── api/  # API interaction layer
│   │   ├── wq_client.py  # WorldQuant API client
│   │   └── ai_client.py  # AI model interface
│   ├── core/  # Business logic
│   │   ├── alpha_generator.py  # Alpha creation
│   │   ├── alpha_polisher.py  # Alpha refinement
│   │   ├── alpha_simulator.py  # Simulation management
│   │   └── alpha_submitter.py  # Submission handling
│   ├── models/  # Data models
│   │   └── alpha.py  # Alpha data classes
│   └── utils/  # Shared utilities
│       ├── config.py  # Configuration management
│       ├── logging.py  # Centralized logging
│       └── validators.py  # Input validation
├── scripts/  # Command-line tools
│   ├── generate_alphas.py
│   ├── polish_alphas.py
│   ├── mine_expressions.py
│   └── submit_alphas.py
└── tests/  # Unit/integration tests
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Run tests: `pytest`
5. Commit your changes: `git commit -am 'Add some feature'`
6. Push to the branch: `git push origin feature/your-feature-name`
7. Submit a pull request

## License

MIT License