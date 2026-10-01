"""路线适配器的最小描述。"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class CampaignRoute:
    name: str
    enumerate_rewrites: Callable[[object], Sequence[Mapping[str, object]]]
    apply_hint: Callable[..., object]
    rewrite_realizes: Callable[[object, object, Mapping[str, object]], bool]
    build_profile: Callable[..., object]
    profile_actions: Callable[[object, object], Sequence[Mapping[str, object]]]
    apply_profile: Callable[..., object]
    profile_realizes: Callable[[object, object, Mapping[str, object]], bool]
