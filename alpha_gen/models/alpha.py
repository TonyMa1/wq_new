"""
Data models for WorldQuant alphas and related entities.
Provides structured representation of alphas, simulations, and results.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import json
import re

class ValidationError(Exception):
    """Raised when alpha validation fails."""
    pass

@dataclass
class SimulationSettings:
    """Simulation settings for WorldQuant Brain."""
    
    instrument_type: str = 'EQUITY'
    region: str = 'USA'
    universe: str = 'TOP3000'
    delay: int = 1
    decay: int = 0
    neutralization: str = 'INDUSTRY'
    truncation: float = 0.08
    pasteurization: str = 'ON'
    unit_handling: str = 'VERIFY'
    nan_handling: str = 'OFF'
    language: str = 'FASTEXPR'
    visualization: bool = False
    
    def to_api_format(self) -> Dict[str, Any]:
        """Convert to API-compatible dictionary format."""
        return {
            'instrumentType': self.instrument_type,
            'region': self.region,
            'universe': self.universe,
            'delay': self.delay,
            'decay': self.decay,
            'neutralization': self.neutralization,
            'truncation': self.truncation,
            'pasteurization': self.pasteurization,
            'unitHandling': self.unit_handling,
            'nanHandling': self.nan_handling,
            'language': self.language,
            'visualization': self.visualization,
        }
    
    @classmethod
    def from_api_format(cls, data: Dict[str, Any]) -> 'SimulationSettings':
        """Create settings object from API response."""
        return cls(
            instrument_type=data.get('instrumentType', 'EQUITY'),
            region=data.get('region', 'USA'),
            universe=data.get('universe', 'TOP3000'),
            delay=data.get('delay', 1),
            decay=data.get('decay', 0),
            neutralization=data.get('neutralization', 'INDUSTRY'),
            truncation=data.get('truncation', 0.08),
            pasteurization=data.get('pasteurization', 'ON'),
            unit_handling=data.get('unitHandling', 'VERIFY'),
            nan_handling=data.get('nanHandling', 'OFF'),
            language=data.get('language', 'FASTEXPR'),
            visualization=data.get('visualization', False),
        )

@dataclass
class AlphaCheck:
    """Result of a specific alpha check."""
    
    name: str
    result: str
    limit: Optional[float] = None
    value: Optional[float] = None
    
    @property
    def passed(self) -> bool:
        """Check if this check passed."""
        return self.result == "PASS"
    
    @classmethod
    def from_api_format(cls, data: Dict[str, Any]) -> 'AlphaCheck':
        """Create check object from API response."""
        return cls(
            name=data.get('name', ''),
            result=data.get('result', ''),
            limit=data.get('limit'),
            value=data.get('value')
        )

@dataclass
class AlphaMetrics:
    """Performance metrics for an alpha."""
    
    sharpe: float = 0.0
    fitness: Optional[float] = None
    turnover: float = 0.0
    returns: float = 0.0
    drawdown: float = 0.0
    margin: float = 0.0
    long_count: int = 0
    short_count: int = 0
    checks: List[AlphaCheck] = field(default_factory=list)
    
    @property
    def passed_checks(self) -> bool:
        """Check if all checks passed."""
        return all(check.passed for check in self.checks)
    
    @property
    def total_positions(self) -> int:
        """Get total position count."""
        return self.long_count + self.short_count
    
    @classmethod
    def from_api_format(cls, data: Dict[str, Any]) -> 'AlphaMetrics':
        """Create metrics object from API response."""
        checks = [
            AlphaCheck.from_api_format(check) 
            for check in data.get('checks', [])
        ]
        
        return cls(
            sharpe=data.get('sharpe', 0.0),
            fitness=data.get('fitness'),
            turnover=data.get('turnover', 0.0),
            returns=data.get('returns', 0.0),
            drawdown=data.get('drawdown', 0.0),
            margin=data.get('margin', 0.0),
            long_count=data.get('longCount', 0),
            short_count=data.get('shortCount', 0),
            checks=checks
        )

@dataclass
class Alpha:
    """Representation of a WorldQuant alpha."""
    
    expression: str
    id: Optional[str] = None
    name: Optional[str] = None
    settings: SimulationSettings = field(default_factory=SimulationSettings)
    metrics: Optional[AlphaMetrics] = None
    date_created: Optional[datetime] = None
    date_submitted: Optional[datetime] = None
    status: str = "DRAFT"
    grade: str = "UNKNOWN"
    tags: List[str] = field(default_factory=list)
    color: Optional[str] = None
    description: Optional[str] = None
    
    def __post_init__(self):
        """Validate the alpha after initialization."""
        self.validate()
    
    def validate(self) -> bool:
        """
        Validate the alpha expression for syntax errors.
        
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        # Basic syntax checks
        expression = self.expression.strip()
        
        # Check for balanced parentheses
        if expression.count('(') != expression.count(')'):
            raise ValidationError("Unbalanced parentheses in expression")
        
        # Check for common syntax errors
        if re.match(r'^\d+\.?$|^[a-zA-Z]+$', expression):
            raise ValidationError("Expression is too simple (just a number or word)")
        
        # Check for function calls
        if not re.search(r'[a-zA-Z_][a-zA-Z0-9_]*\s*\(', expression):
            raise ValidationError("No function calls found in expression")
        
        return True
    
    @classmethod
    def from_api_format(cls, data: Dict[str, Any]) -> 'Alpha':
        """Create alpha object from API response."""
        # Parse dates
        date_created = None
        if 'dateCreated' in data:
            try:
                date_created = datetime.fromisoformat(data['dateCreated'].replace('Z', '+00:00'))
            except (ValueError, TypeError):
                pass
        
        date_submitted = None
        if 'dateSubmitted' in data and data['dateSubmitted']:
            try:
                date_submitted = datetime.fromisoformat(data['dateSubmitted'].replace('Z', '+00:00'))
            except (ValueError, TypeError):
                pass
        
        # Extract expression
        expression = ""
        if 'regular' in data and isinstance(data['regular'], dict) and 'code' in data['regular']:
            expression = data['regular']['code']
        
        # Extract description
        description = None
        if 'regular' in data and isinstance(data['regular'], dict) and 'description' in data['regular']:
            description = data['regular']['description']
        
        # Create settings
        settings = SimulationSettings.from_api_format(data.get('settings', {}))
        
        # Create metrics
        metrics = None
        if 'is' in data and data['is']:
            metrics = AlphaMetrics.from_api_format(data['is'])
        
        return cls(
            expression=expression,
            id=data.get('id'),
            name=data.get('name'),
            settings=settings,
            metrics=metrics,
            date_created=date_created,
            date_submitted=date_submitted,
            status=data.get('status', 'DRAFT'),
            grade=data.get('grade', 'UNKNOWN'),
            tags=data.get('tags', []),
            color=data.get('color'),
            description=description
        )
    
    def to_api_format(self) -> Dict[str, Any]:
        """Convert to API-compatible dictionary format."""
        return {
            'type': 'REGULAR',
            'settings': self.settings.to_api_format(),
            'regular': self.expression
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        def _serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Type {type(obj)} not serializable")
        
        data = {
            'expression': self.expression,
            'id': self.id,
            'name': self.name,
            'settings': self.settings.to_api_format(),
            'metrics': {
                'sharpe': self.metrics.sharpe,
                'fitness': self.metrics.fitness,
                'turnover': self.metrics.turnover,
                'returns': self.metrics.returns,
                'drawdown': self.metrics.drawdown,
                'margin': self.metrics.margin,
                'long_count': self.metrics.long_count,
                'short_count': self.metrics.short_count,
            } if self.metrics else None,
            'date_created': self.date_created,
            'date_submitted': self.date_submitted,
            'status': self.status,
            'grade': self.grade,
            'tags': self.tags,
            'color': self.color,
            'description': self.description,
        }
        
        return json.dumps(data, default=_serializer, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Alpha':
        """Create alpha object from JSON string."""
        data = json.loads(json_str)
        
        # Convert dates from strings
        date_created = None
        if 'date_created' in data and data['date_created']:
            try:
                date_created = datetime.fromisoformat(data['date_created'])
            except (ValueError, TypeError):
                pass
        
        date_submitted = None
        if 'date_submitted' in data and data['date_submitted']:
            try:
                date_submitted = datetime.fromisoformat(data['date_submitted'])
            except (ValueError, TypeError):
                pass
        
        # Create settings
        settings = SimulationSettings.from_api_format(data.get('settings', {}))
        
        # Create metrics
        metrics = None
        if 'metrics' in data and data['metrics']:
            metrics_data = data['metrics']
            checks = []
            
            metrics = AlphaMetrics(
                sharpe=metrics_data.get('sharpe', 0.0),
                fitness=metrics_data.get('fitness'),
                turnover=metrics_data.get('turnover', 0.0),
                returns=metrics_data.get('returns', 0.0),
                drawdown=metrics_data.get('drawdown', 0.0),
                margin=metrics_data.get('margin', 0.0),
                long_count=metrics_data.get('long_count', 0),
                short_count=metrics_data.get('short_count', 0),
                checks=checks
            )
        
        return cls(
            expression=data.get('expression', ''),
            id=data.get('id'),
            name=data.get('name'),
            settings=settings,
            metrics=metrics,
            date_created=date_created,
            date_submitted=date_submitted,
            status=data.get('status', 'DRAFT'),
            grade=data.get('grade', 'UNKNOWN'),
            tags=data.get('tags', []),
            color=data.get('color'),
            description=data.get('description')
        )

@dataclass
class SimulationResult:
    """Result of a WorldQuant simulation."""
    
    alpha: Optional[Alpha] = None
    simulation_id: Optional[str] = None
    status: str = "PENDING"
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    @classmethod
    def from_api_format(cls, data: Dict[str, Any], expression: str) -> 'SimulationResult':
        """Create simulation result object from API response."""
        status = data.get('status', 'PENDING')
        error_message = data.get('message')
        simulation_id = data.get('id')
        
        alpha = None
        if status == 'COMPLETE' and 'alpha' in data:
            # Just store the alpha ID for now
            alpha = Alpha(expression=expression, id=data['alpha'])
        
        return cls(
            alpha=alpha,
            simulation_id=simulation_id,
            status=status,
            error_message=error_message
        )