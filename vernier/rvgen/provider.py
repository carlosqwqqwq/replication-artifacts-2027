"""最小的 RISC-V-DV 程序接入对象。"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from framework._util import is_riscv_elf


@dataclass(frozen=True)
class CaseProgram:
    """下游共享的最小输入：完整程序路径和必要运行参数。"""

    program: Path
    run_params: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        path = Path(self.program).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix.lower() != ".s" and not is_riscv_elf(path):
            raise ValueError("program must be a .S source or ELF file")
        if not isinstance(self.run_params, Mapping):
            raise ValueError("run_params must be an object")
        object.__setattr__(self, "program", path)
        object.__setattr__(self, "run_params", dict(self.run_params))
