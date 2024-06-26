from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union
from torch import Tensor

Params = Union[Iterable[Tensor], Iterable[Dict[str, Any]]]
LossClosure = Callable[[], float]
OptLossClosure = Optional[LossClosure]
Betas2 = Tuple[float, float]
Truncated = Tuple[bool, float]
State = Dict[str, Any]
OptFloat = Optional[float]