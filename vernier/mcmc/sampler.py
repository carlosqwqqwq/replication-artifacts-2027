"""最小单链 Metropolis-Hastings 采样器。"""

import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

REJECTED = object()


def _checked_log(value: object, name: str, *, proposal: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是有限值或负无穷")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} 必须是有限值或负无穷") from exc
    if math.isnan(result) or result == math.inf or proposal and result > 0.0:
        raise ValueError(f"{name} 必须是有限值或负无穷")
    return result


@dataclass(frozen=True)
class MHProposal:
    candidate: Any
    log_q_forward: float | None = None
    log_q_reverse: float | None = None
    q_status: str | None = None


@dataclass(frozen=True)
class MHRun:
    """一次 MH 运行的核心结果。"""

    samples: tuple
    mh_attempts: int = 0


def mh_log_alpha(
    log_pi_current: float, log_pi_candidate: float,
    log_q_forward: float | None = None, log_q_reverse: float | None = None,
) -> float:
    log_pi_current = _checked_log(log_pi_current, "log density")
    log_pi_candidate = _checked_log(log_pi_candidate, "log density")
    if (log_q_forward is None) != (log_q_reverse is None):
        raise ValueError("forward and reverse proposal probabilities must be paired")
    if log_q_forward is not None:
        log_q_forward = _checked_log(log_q_forward, "log q forward", proposal=True)
        log_q_reverse = _checked_log(log_q_reverse, "log q reverse", proposal=True)
        if log_q_forward == -math.inf and log_q_reverse == -math.inf:
            raise ValueError("forward and reverse proposal probabilities cannot both be zero")
        if log_q_forward == -math.inf:
            raise ValueError("forward proposal probability cannot be zero")
    if log_pi_current == -math.inf and log_pi_candidate > -math.inf:
        value = math.inf
    elif log_pi_current == -math.inf or log_pi_candidate == -math.inf:
        value = -math.inf
    else:
        value = log_pi_candidate - log_pi_current
    if log_q_forward is not None:
        value = (
            -math.inf if log_q_reverse == -math.inf
            else value + log_q_reverse - log_q_forward
        )
    return min(0.0, value)


def mh_chain(
    rng: random.Random,
    current: Any,
    proposal: Callable[[random.Random, Any], MHProposal],
    *,
    chain_len: int,
    on_step: Callable[[dict[str, Any]], bool | None] | None = None,
    on_accept: Callable[[Any], bool | None] | None = None,
    log_density: Callable[[Any], float],
    max_seconds: float | None = None,
) -> MHRun:
    """单条 MH 链。

    参数:
      rng: 链专属随机源；
      current: 初始状态；
      proposal: proposal(rng, current) -> ``MHProposal``；返回
        ``REJECTED`` 时跳过接受判断并保留 current；
      chain_len: 样本步数；
    返回样本和 MH 尝试次数。
    """
    if type(chain_len) is not int or chain_len < 0:
        raise ValueError("chain_len must be a non-negative integer")
    samples: list = []
    mh_attempts = 0
    started = time.monotonic()

    current_log = _checked_log(log_density(current), "log density") if chain_len else -math.inf
    for step in range(chain_len):
        # ponytail: 墙钟预算按剩余时间收尾；固定 steps 在长窗口才跑满。
        if max_seconds is not None and time.monotonic() - started >= max_seconds:
            break
        proposed = proposal(rng, current)
        if proposed is REJECTED:
            proposed = MHProposal(REJECTED)
        elif not isinstance(proposed, MHProposal):
            raise ValueError("proposal must return MHProposal or REJECTED")
        nxt = proposed.candidate
        previous_log = current_log
        checked_forward = None if proposed.log_q_forward is None else _checked_log(
            proposed.log_q_forward, "log_q_forward", proposal=True
        )
        checked_reverse = None if proposed.log_q_reverse is None else _checked_log(
            proposed.log_q_reverse, "log_q_reverse", proposal=True
        )
        if (checked_forward is None) != (checked_reverse is None):
            raise ValueError("forward and reverse proposal probabilities must be paired")
        has_q = checked_forward is not None and checked_reverse is not None
        if proposed.q_status is not None and (
                not isinstance(proposed.q_status, str)
                or proposed.q_status != "exact"
        ):
            raise ValueError("q_status must be exact")
        q_status = proposed.q_status or "exact"
        if q_status == "exact" and not has_q:
            if nxt is not REJECTED:
                raise ValueError("exact q requires both forward and reverse values")
        if nxt is REJECTED:
            samples.append(current)
            if on_step is not None and on_step({
                    "step": step,
                    "accepted": False,
                    "alpha": None,
                    "log_pi_current": previous_log,
                    "log_pi_candidate": None,
                    "log_alpha": None,
                    "log_q_forward": None,
                    "log_q_reverse": None,
                    "q_status": q_status,
                    "mh_attempt": False,
                    "mh_accepted": False,
                    "state_accepted": False,
                    "accept_draw": None,
                }) is True:
                break
            continue
        mh_attempts += 1
        next_log = _checked_log(log_density(nxt), "log density")
        log_q_forward = checked_forward if has_q else None
        log_q_reverse = checked_reverse if has_q else None
        log_alpha = mh_log_alpha(
            previous_log, next_log, log_q_forward, log_q_reverse,
        )
        alpha = 1.0 if log_alpha >= 0.0 else math.exp(log_alpha)
        accept_draw = rng.random()
        if not 0 <= accept_draw < 1:
            raise ValueError("rng.random() must return a value in [0, 1)")
        draw_accepted = accepted_step = accept_draw < alpha
        if accepted_step:
            if on_accept is not None and on_accept(nxt) is False:
                accepted_step = False
                current_log = previous_log
            else:
                current = nxt
                current_log = next_log
        samples.append(current)
        if on_step is not None and on_step({
                "step": step,
                "accepted": accepted_step,
                "mh_accepted": draw_accepted,
                "alpha": alpha,
                "log_pi_current": previous_log,
                "log_pi_candidate": next_log,
                "log_alpha": log_alpha,
                "log_q_forward": log_q_forward,
                "log_q_reverse": log_q_reverse,
                "q_status": q_status,
                "mh_attempt": True,
                "state_accepted": accepted_step,
                "accept_draw": accept_draw,
            }) is True:
                break
    return MHRun(samples=tuple(samples), mh_attempts=mh_attempts)
