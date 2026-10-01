"""为裸机 RISC-V-DV source 提供 Runner context 的 RVOBS1 观察。"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import math
from collections.abc import Mapping
from pathlib import Path

from framework.direct_case import OBSERVATION_HEADER_SIZE, OBSERVATION_MAGIC
from framework.direct_elf import _toolchain_profile
from framework.spec_definedness import enabled_extensions, mabi_matches_isa
try:
    from framework.rvemi.program import _expand_asm_macros, _register_index
except ImportError:
    _REGISTER_ALIASES = dict(zip(
        ("zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2", "s0", "s1",
         "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "s2", "s3", "s4",
         "s5", "s6", "s7", "s8", "s9", "s10", "s11", "t3", "t4", "t5", "t6"),
        range(32),
    ))
    _REGISTER_ALIASES["fp"] = 8
    def _register_index(value: object) -> int | None:
        if not isinstance(value, str):
            return None
        if value in _REGISTER_ALIASES:
            return _REGISTER_ALIASES[value]
        return int(value[1:]) if value.startswith("x") and value[1:].isdigit() and int(value[1:]) < 32 else None
    def _expand_asm_macros(source: str, *, line_directives: bool = True) -> str:
        return source


OBSERVATION_SIZE = OBSERVATION_HEADER_SIZE
_OBSERVATION_MAGIC = int.from_bytes(OBSERVATION_MAGIC, "little")
_INITIAL_STATE_META = frozenset(("entry", "privilege", "registers"))

_RVDV_USER_SYSTEM_MNEMONICS = frozenset((
    "csrr", "csrrw", "csrrs", "csrrc", "csrrwi", "csrrsi", "csrrci",
    "csrw", "csrs", "csrc", "csrwi", "csrsi", "csrci",
    "mret", "sret", "uret", "wfi", "sfence.vma", "hfence.gvma",
    "hfence.vvma", "ebreak",
))
_RVDV_USER_MEMORY_MNEMONICS = frozenset((
    "lb", "lbu", "lh", "lhu", "lw", "lwu", "ld",
    "sb", "sh", "sw", "sd", "c.lw", "c.ld", "c.sw", "c.sd",
))
_RVDV_USER_SYSTEM_LINE = re.compile(
    r"^(?P<indent>\s*)(?:(?P<label>(?:[0-9]+|[.$A-Za-z_][\w.$]*):)\s*)?"
    r"(?P<mnemonic>[A-Za-z][A-Za-z0-9.]*)\b(?P<rest>.*)$",
    re.IGNORECASE,
)
_RVDV_USER_MEMORY_BASE = re.compile(
    r"\(\s*(?:x(?:[0-9]|[12][0-9]|3[01])|zero|ra|sp|gp|tp|t[0-6]|"
    r"s(?:[0-9]|1[01])|a[0-7]|fp)\s*\)", re.IGNORECASE,
)
def _neutralize_rvdv_user_instructions(source: str) -> str:
    """将 RISC-V-DV 机器态系统指令变成 Linux 用户态安全的空操作。

    RISC-V-DV 的裸机 frame 会把 CSR/特权指令、随机栈写入和随机内存基址
    散布到程序和 trap handler 中。Linux-user 没有机器态 trap 环境，且
    不能访问裸机地址；用同宽的普通整数指令、固定用户栈基址和栈内存基址
    保留源文件行数，避免为 Linux 适配引入第二套生成参数。
    """
    trailing_newline = source.endswith("\n")
    converted: list[str] = []
    source_lines = source.splitlines()

    def previous_code_line(index: int) -> str | None:
        for previous in reversed(source_lines[:index]):
            stripped = previous.strip()
            if stripped and not stripped.startswith(("#", ".")):
                return previous
        return None

    def is_return_setup(index: int, register: int | None) -> bool:
        if register is None:
            return False
        previous = previous_code_line(index)
        if previous is None:
            return False
        setup = _RVDV_USER_SYSTEM_LINE.match(previous)
        if setup is None or setup.group("mnemonic").lower() != "addi":
            return False
        setup_operands = [
            part.strip().lower()
            for part in setup.group("rest").split("#", 1)[0].split(",")
            if part.strip()
        ]
        return (
            len(setup_operands) == 3
            and _register_index(setup_operands[0]) == register
            and setup_operands[2] == "0"
            and _register_index(setup_operands[1]) is not None
        )

    def is_test_done_setup(index: int, register: int | None) -> bool:
        if register is None:
            return False
        previous = previous_code_line(index)
        if previous is None:
            return False
        setup = _RVDV_USER_SYSTEM_LINE.match(previous)
        if setup is None or setup.group("mnemonic").lower() != "la":
            return False
        setup_operands = [
            part.strip().lower()
            for part in setup.group("rest").split("#", 1)[0].split(",")
            if part.strip()
        ]
        return (
            len(setup_operands) == 2
            and _register_index(setup_operands[0]) == register
            and setup_operands[1] == "test_done"
        )

    for index, line in enumerate(source_lines):
        match = _RVDV_USER_SYSTEM_LINE.match(line)
        if match is None:
            converted.append(line)
            continue
        mnemonic = match.group("mnemonic").lower()
        rest = match.group("rest")
        prefix = match.group("indent") + (
            f"{match.group('label')} " if match.group("label") else ""
        )
        code, marker, comment = rest.partition("#")
        operands = [part.strip() for part in code.split(",") if part.strip()]
        if (mnemonic.startswith("b") or mnemonic.startswith("c.b")) and operands \
                and re.fullmatch(r"[0-9]+b", operands[-1]):
            replacement = "c.nop" if mnemonic.startswith("c.") else "nop"
            converted.append(f"{prefix}{replacement}" + (f" # {comment.lstrip()}" if marker else ""))
            continue
        if mnemonic == "jal" and operands \
                and re.fullmatch(r"sub_[0-9]+", operands[-1], re.IGNORECASE):
            # A DV subprogram may conditionally skip its stack save and still
            # execute the matching restore/return.  In the Linux-user
            # projection that makes the indirect return depend on an
            # uninitialised user-stack word.  Keep the generated body and its
            # source mapping, but skip calls into those machine-mode helper
            # subprograms; the main frame still reaches test_done.
            converted.append(f"{prefix}nop" + (f" # {comment.lstrip()}" if marker else ""))
            continue
        if (
            mnemonic == "la" and operands and _register_index(operands[0].lower()) == 2
            and re.search(r"\buser_stack_end\b", code, re.IGNORECASE)
        ):
            replacement = f"{prefix}la {operands[0]}, user_stack_end-256"
            if marker:
                replacement += f" # {comment.lstrip()}"
            converted.append(replacement)
            continue
        if mnemonic == "la":
            converted.append(line)
            continue
        if mnemonic in {"jalr", "c.jr", "c.jalr"}:
            # RISC-V-DV emits real returns as ``addi r, ra, 0`` followed by
            # ``jalr r, r, 0`` (or ``c.jr r``), and the frame exits through
            # ``la r, test_done`` followed by ``jalr x0, r, 0``.  Other
            # indirect jumps use random register contents and cannot be
            # executed faithfully in a Linux-user projection; retaining one
            # would jump into an arbitrary address and turn an observation
            # into a guest crash.
            keep = False
            if mnemonic == "jalr" and len(operands) >= 2:
                destination = _register_index(operands[0].lower())
                base = _register_index(operands[1].lower())
                keep = (
                    destination == base and is_return_setup(index, base)
                ) or (
                    destination == 0 and is_test_done_setup(index, base)
                )
            elif operands:
                base = _register_index(operands[0].lower())
                keep = is_return_setup(index, base)
            if keep:
                converted.append(line)
            else:
                comment_suffix = f" # {comment.lstrip()}" if marker else ""
                replacement = "c.nop" if mnemonic.startswith("c.") else "nop"
                converted.append(f"{prefix}{replacement}{comment_suffix}")
            continue
        if mnemonic in _RVDV_USER_MEMORY_MNEMONICS:
            code = _RVDV_USER_MEMORY_BASE.sub("(sp)", code)
            suffix = f"#{comment}" if marker else ""
            converted.append(f"{prefix}{mnemonic}{code}{suffix}")
            continue
        if mnemonic not in _RVDV_USER_SYSTEM_MNEMONICS:
            converted.append(line)
            continue
        if mnemonic.startswith("csrr") and operands:
            destination = operands[0].lower()
            if _register_index(destination) is None:
                destination = "x0"
        else:
            destination = "x0"
        replacement = f"{prefix}or {destination}, {destination}, zero"
        if marker:
            replacement += f" # {comment.lstrip()}"
        converted.append(replacement)
    result = "\n".join(converted)
    return result + ("\n" if trailing_newline else "")


def _fast_copy_loop(source: str, xlen: int) -> str:
    width = 8 if xlen == 64 else 4
    load, store = ("ld", "sd") if xlen == 64 else ("lw", "sw")
    for start, end, byte, dest, temp in (
        ("x31", "x30", "x29", "x28", "x27"),
        ("x31", "x30", "x29", "x26", "x28"),
        ("t6", "t5", "t4", "t3", "t2"),
    ):
        old = "\n".join((
            "rvobs_copy_loop:",
            f"  beq {start}, {end}, rvobs_copy_done",
            f"  lbu {byte}, 0({start})",
            f"  sb {byte}, 0({dest})",
            f"  addi {start}, {start}, 1",
            f"  addi {dest}, {dest}, 1",
            "  j rvobs_copy_loop",
        ))
        new = "\n".join((
            "rvobs_copy_loop:",
            f"  beq {start}, {end}, rvobs_copy_done",
            f"  sub {byte}, {end}, {start}",
            f"  li {temp}, {width}",
            f"  bltu {byte}, {temp}, rvobs_copy_tail",
            f"  {load} {byte}, 0({start})",
            f"  {store} {byte}, 0({dest})",
            f"  addi {start}, {start}, {width}",
            f"  addi {dest}, {dest}, {width}",
            "  j rvobs_copy_loop",
            "rvobs_copy_tail:",
            f"  beq {start}, {end}, rvobs_copy_done",
            f"  lbu {byte}, 0({start})",
            f"  sb {byte}, 0({dest})",
            f"  addi {start}, {start}, 1",
            f"  addi {dest}, {dest}, 1",
            "  j rvobs_copy_tail",
        ))
        source = source.replace(old, new, 1)
    return source


def _initial_state_setup(
    initial_state: Mapping[str, object] | None, *, xlen: int, stack_register: int,
) -> list[str]:
    if not initial_state:
        return []
    if initial_state.get("entry") not in (None, "_start"):
        raise RuntimeError("initial-state-entry-not-materialized")
    registers = initial_state.get("registers", {})
    if not isinstance(registers, Mapping):
        raise ValueError("initial_state.registers must be an object")
    values: dict[int, object] = {}
    for key, value in initial_state.items():
        index = _register_index(str(key).lower())
        if index is not None:
            values[index] = value
        elif key not in {"entry", "privilege", "registers"}:
            raise RuntimeError("initial-state-not-materialized")
    for key, value in registers.items():
        index = _register_index(str(key).lower())
        if index is None:
            raise RuntimeError("initial-state-register-not-materialized")
        values[index] = value
    lines = []
    for index, value in sorted(values.items()):
        if not isinstance(value, str) and type(value) is not int:
            raise ValueError("initial-state-register-value-invalid")
        try:
            value = int(value, 0) if isinstance(value, str) else int(value)
        except (TypeError, ValueError) as error:
            raise ValueError("initial-state-register-value-invalid") from error
        if index == 0:
            if value:
                raise ValueError("initial-state-x0-must-be-zero")
            continue
        if index == stack_register:
            raise RuntimeError("initial-state-stack-register-reserved")
        value &= (1 << xlen) - 1
        if value >= 1 << (xlen - 1):
            value -= 1 << xlen
        lines.append(f"  li x{index}, {value}")
    return lines


def linux_user_harness(
    source: str, *, xlen: int = 64, initial_state: Mapping[str, object] | None = None,
    prologue_lines: tuple[str, ...] = (),
) -> str:
    """将 DV 的裸机 tohost 终止改成 Linux RVOBS1 输出。"""
    if xlen not in {32, 64}:
        raise ValueError("xlen must be 32 or 64")
    if any(re.fullmatch(r"\s*h[1-9]\d*_start:\s*", line) for line in source.splitlines()):
        raise ValueError("multi-hart source is unsupported")
    if re.search(r'(?m)^[ \t]*(?:\.include|#\s*include)[ \t]+"user_init\.s"[^\r\n]*(?:\r?\n|$)', source):
        raise ValueError("user_init.s must be expanded before harness conversion")
    lines = source.splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip() == "_start:"), None)
    init_label = "init"
    init = next((i for i, line in enumerate(lines) if line.strip() == "init:"), None)
    if init is None:
        init_label = "h0_start"
        init = next((i for i, line in enumerate(lines) if line.strip() == "h0_start:"), None)
    done = next((i for i, line in enumerate(lines) if line.strip() == "test_done:"), None)
    tohost = next((i for i, line in enumerate(lines) if line.strip() == "write_tohost:"), None)
    if start is None or init is None or done is None or tohost is None or not start < init <= done < tohost:
        raise ValueError("RISC-V-DV source has no supported bare-metal frame")
    kernel_stack_setup = []
    for line in lines[:init]:
        fields = line.split("#", 1)[0].replace(",", " ").split()
        if (
            len(fields) >= 3 and fields[0].lower() == "la"
            and _register_index(fields[1]) not in (None, 0, 2)
            and re.fullmatch(
                r"kernel_stack_end(?:[+-](?:0x[0-9a-fA-F]+|[0-9]+))?",
                fields[2],
            )
        ):
            # RISC-V-DV leaves cfg.tp out of init_gpr because its bare-metal
            # entry prefix initializes it to the kernel stack.  Linux-user
            # skips that prefix, so materialize the same setup after init_gpr.
            kernel_stack_setup.append(line)
    logical, logical_lines = 1, []
    for raw in lines:
        directive = re.fullmatch(r"#\s*(?:line\s+)?(\d+)(?:\s+.*)?", raw.strip(), re.I)
        logical_lines.append(None if directive else logical)
        if directive:
            logical = int(directive[1])
        else:
            logical += 1
    stack_line, stack_register = next((
        (i, int(fields[1][1:] if fields[1].startswith("x") else "2"))
        for i, line in enumerate(lines[init:done], init)
        if len(fields := line.replace(",", " ").split()) >= 3
        and fields[0] == "la"
        and (fields[1].startswith("x") or fields[1] == "sp")
        and (fields[1][1:] if fields[1].startswith("x") else "2").isdigit()
        and re.search(
            r"user_stack_end(?:[+-](?:0x[0-9a-fA-F]+|[0-9]+))?$",
            fields[2],
        )
    ), (None, None))
    if stack_register is None:
        raise ValueError("RISC-V-DV source has no initialized user stack")
    initial_setup = _initial_state_setup(
        initial_state, xlen=xlen, stack_register=stack_register,
    )
    pointer_register = 5 if stack_register != 5 else 27
    temp_register = 27 if 27 not in {stack_register, pointer_register} else 28
    buffer_register = 30 if pointer_register == 27 else pointer_register
    subprogram_start = next(
        (i for i in range(done + 1, tohost)
         if re.match(r"\s*sub_[0-9]+:\s*", lines[i])),
        tohost,
    )
    main = next((i for i in range(init + 1, done) if lines[i].strip() == "main:"), None)
    setup_at = main if main is not None else stack_line + 1
    body = lines[: start + 1] + [
        f"#line {logical_lines[start] or start + 1}", "test_start:",
        # Linux-user startup does not initialize the ELF global pointer.  GCC
        # emits small-data references through ``gp`` (CSmith uses them), so
        # leaving it at zero makes otherwise valid guests fault before the
        # observer checkpoint.
        "  lla gp, __global_pointer$",
        f"  j {init_label}", f"#line {logical_lines[init] or init + 1}",
    ] + lines[init:setup_at] + kernel_stack_setup + list(prologue_lines) + initial_setup + [
        "rvgen_program_body_start:",
        f"#line {logical_lines[setup_at] or setup_at + 1}",
    ] + lines[setup_at:done + 1] + ["  j exit_checkpoint"] \
        + lines[subprogram_start:tohost] \
        + ["rvgen_program_body_end:"]
    body += [
        "exit_checkpoint:", "  j emit_observation",
        f"#line {logical_lines[tohost] or tohost + 1}",
    ] + lines[tohost:]
    source = "\n".join(body) + ("\n" if source.endswith(("\n", "\r")) else "")
    source = re.sub(
        r'(?m)^[ \t]*(?:\.include|#\s*include)[ \t]+"user_define\.h"[ \t]*(?:;[ \t]*)?',
        "", source,
    )
    data_section = re.search(
        r"(?m)^\.section[ \t]+\.data(?:[ \t]*,[^\r\n]*)?[ \t]*(?:\r?\n|$)",
        source,
    )
    if data_section is None:
        raise ValueError("RISC-V-DV source has no data section")
    text, data = source[:data_section.start()], source[data_section.end():]
    data = re.sub(
        r"(?ms)^ecall_handler:[^\n]*\n.*?(?=^illegal_instr_handler:)",
        "ecall_handler:\n  csrr x9, 0x341\n  addi x9, x9, 4\n  csrw 0x341, x9\n  mret\n\n",
        data,
        count=1,
    )
    store = "sd" if xlen == 64 else "sw"

    def snapshot_store(register: int, offset: int) -> list[str]:
        return [f"  {store} x{register}, {offset}(x{pointer_register})"] + (
            [f"  sw zero, {offset + 4}(x{pointer_register})"] if xlen == 32 else []
        )

    saved_pointer_offset = -16 if xlen == 64 else -8
    saved_temp_offset = -8 if xlen == 64 else -4
    load = "ld" if xlen == 64 else "lw"
    scratch = iter(
        register for register in range(31, 0, -1)
        if register not in {stack_register, pointer_register, temp_register, buffer_register}
    )
    memory_start, memory_end, memory_byte, memory_dest = (next(scratch) for _ in range(4))
    magic = [
        f"  li x{temp_register}, {_OBSERVATION_MAGIC}",
        f"  {store} x{temp_register}, 0(x{pointer_register})",
    ] if xlen == 64 else [
        f"  li x{temp_register}, {_OBSERVATION_MAGIC & 0xffffffff}",
        f"  sw x{temp_register}, 0(x{pointer_register})",
        f"  li x{temp_register}, {_OBSERVATION_MAGIC >> 32}",
        f"  sw x{temp_register}, 4(x{pointer_register})",
    ]
    observer = [
        ".option push",
        ".option norelax",
        ".option norvc",
        "emit_observation:",
        f"  {store} x{pointer_register}, {saved_pointer_offset}(x{stack_register})",
        f"  {store} x{temp_register}, {saved_temp_offset}(x{stack_register})",
        f"  la x{pointer_register}, obs_buf",
        *sum((snapshot_store(index, 16 + index * 8) for index in range(32)), []),
        f"  {load} x{temp_register}, {saved_pointer_offset}(x{stack_register})",
        f"  {store} x{temp_register}, {16 + pointer_register * 8}(x{pointer_register})",
        f"  {load} x{temp_register}, {saved_temp_offset}(x{stack_register})",
        f"  {store} x{temp_register}, {16 + temp_register * 8}(x{pointer_register})",
        *magic,
        f"  lla x{temp_register}, exit_checkpoint",
        f"  {store} x{temp_register}, 8(x{pointer_register})",
        *(
            [f"  mv x{buffer_register}, x{pointer_register}"]
            if buffer_register != pointer_register
            else []
        ),
        *([f"  sw zero, 12(x{pointer_register})"] if xlen == 32 else []),
        f"  lla x{memory_start}, test_memory_start",
        f"  lla x{memory_end}, test_memory_end",
        f"  sub x{memory_byte}, x{memory_end}, x{memory_start}",
        f"  {store} x{memory_byte}, 272(x{buffer_register})",
        *([f"  sw zero, 276(x{buffer_register})"] if xlen == 32 else []),
        f"  addi x{memory_dest}, x{buffer_register}, {OBSERVATION_SIZE}",
        "rvobs_copy_loop:",
        f"  beq x{memory_start}, x{memory_end}, rvobs_copy_done",
        f"  lbu x{memory_byte}, 0(x{memory_start})",
        f"  sb x{memory_byte}, 0(x{memory_dest})",
        f"  addi x{memory_start}, x{memory_start}, 1",
        f"  addi x{memory_dest}, x{memory_dest}, 1",
        "  j rvobs_copy_loop",
        "rvobs_copy_done:",
        "  li a0, 1",
        f"  mv a1, x{buffer_register}",
        f"  sub a2, x{memory_dest}, x{buffer_register}",
        "  li a7, 64",
        "  ecall",
        "  li a0, 0",
        "  li a7, 93",
        "  ecall",
        "  j .",
        ".option pop",
    ]
    section = re.search(r"(?m)^\.section\s+\.user_stack\b", data)
    if section is None:
        section = re.search(r"(?m)^\.section\s+(?!\.data(?:[\s,]|$))", data)
    data_body, data_tail = (data[:section.start()], data[section.start():]) if section else (data, "")
    memory_labels = tuple(
        re.search(rf"(?m)^[ \t]*{name}:[ \t]*[^\r\n]*$", data_body)
        for name in ("test_memory_start", "test_memory_end")
    )
    if all(memory_labels) and memory_labels[0].start() > memory_labels[1].start():
        raise ValueError("RISC-V-DV data memory range is reversed")
    explicit_memory = all(memory_labels)
    data = (
        f".section .data\n.align 3\nobs_buf:\n.zero {OBSERVATION_SIZE}\n{data_body}{data_tail}"
        if explicit_memory else
        f".section .data\n.align 3\nobs_buf:\n.zero {OBSERVATION_SIZE}\n"
        f"test_memory_start:\n{data_body}\nobserver_scratch:\n.zero 8\n"
        f"test_memory_end:\n{data_tail}"
    )
    return _fast_copy_loop(text + "\n".join(observer) + "\n" + data, xlen)


def _bare_metal_rvdv_harness(
    source: str, *, xlen: int = 64,
) -> str:
    """保留 RVDV 的机器态启动，只替换其 ecall 终止处理。"""
    if xlen not in {32, 64}:
        raise ValueError("xlen must be 32 or 64")
    required = ("_start:", "init:", "test_done:", "write_tohost:",
                "mtvec_handler:", "ecall_handler:", "illegal_instr_handler:")
    if any(not re.search(rf"(?m)^\s*{re.escape(label)}\s*$", source) for label in required):
        raise ValueError("RISC-V-DV source has no supported machine-mode frame")

    body_start = re.search(r"(?m)^main:[^\r\n]*(?:\r?\n|$)", source)
    body_end = re.search(r"(?m)^test_done:\s*$", source)
    if body_start is not None and body_end is not None and body_start.start() < body_end.start():
        source = (
            source[:body_start.start()]
            + _neutralize_rvdv_user_instructions(
                source[body_start.start():body_end.start()]
            )
            + source[body_end.start():]
        )

    # The upstream DV trap prologue swaps one generated register into
    # mscratch, then turns it into the kernel-frame pointer.  Keep that
    # prologue intact and derive both registers from the generated source.
    source, count = re.subn(
        r"(?ms)^ecall_handler:\s*\n.*?(?=^illegal_instr_handler:\s*$)",
        "ecall_handler:\n                  j rvobs_from_trap\n\n",
        source,
        count=1,
    )
    if count != 1:
        raise RuntimeError("RISC-V-DV ecall handler could not be replaced")

    word = xlen // 8
    store = "sd" if xlen == 64 else "sw"
    load = "ld" if xlen == 64 else "lw"
    ecall_at = source.index("ecall_handler:")
    frame_store = re.findall(
        rf"(?m)^\s*{store}\s+x1,\s+{word}\(x([1-9][0-9]?)\)\s*$",
        source[:ecall_at],
    )
    if not frame_store:
        raise ValueError("RISC-V-DV trap frame register could not be inferred")
    frame_register = int(frame_store[-1])
    kernel_store = re.findall(
        rf"(?ms)^\s*csrrw\s+x{frame_register},\s*(?:0x340|mscratch),\s*x{frame_register}\s*$.*?"
        rf"^\s*add\s+x{frame_register},\s*x([1-9][0-9]?),\s*zero\s*$",
        source[:ecall_at],
    )
    if not kernel_store:
        raise ValueError("RISC-V-DV kernel stack register could not be inferred")
    kernel_register = int(kernel_store[-1])
    stack_register = next(
        (
            _register_index(match.group(1).lower())
            for line in source.splitlines()
            if (match := re.match(
                r"^\s*la\s+([A-Za-z][A-Za-z0-9]*)\s*,\s*user_stack_end\s*$",
                line, re.IGNORECASE,
            ))
        ),
        None,
    )
    if stack_register is None:
        raise ValueError("RISC-V-DV user stack register could not be inferred")
    lines = source.splitlines()
    test_done = next(
        (i for i, line in enumerate(lines) if line.strip() == "test_done:"),
        None,
    )
    if test_done is None:
        raise ValueError("RISC-V-DV test_done label could not be located")
    test_done_exit = None
    for index in range(test_done + 1, len(lines)):
        stripped = lines[index].strip()
        if stripped.startswith("sub_") and stripped.endswith(":"):
            break
        if re.match(r"^ecall(?:\s|$)", stripped, re.IGNORECASE):
            test_done_exit = index
            break
    if test_done_exit is None:
        raise ValueError("RISC-V-DV test_done ecall could not be located")
    original_trailing_newline = source.endswith(("\n", "\r"))
    indent = lines[test_done_exit][:len(lines[test_done_exit]) - len(lines[test_done_exit].lstrip())]
    lines[test_done_exit] = f"{indent}j rvobs_test_done"
    source = "\n".join(lines) + ("\n" if original_trailing_newline else "")
    body_start = re.search(
        r"(?m)^\s*main:[^\r\n]*(?:\r?\n|$)", source,
    )
    if body_start is None:
        body_start = re.search(
            r"(?m)^\s*test_done:[^\r\n]*(?:\r?\n|$)", source,
        )
    if body_start is not None and not re.search(
        r"(?m)^\s*rvgen_program_body_start:\s*$", source,
    ):
        source = (
            source[:body_start.start()]
            + "rvgen_program_body_start:\n"
            + source[body_start.start():]
        )
    body_end = re.search(
        r"(?m)^\s*write_tohost:[^\r\n]*(?:\r?\n|$)", source,
    )
    if body_end is not None and not re.search(
        r"(?m)^\s*rvgen_program_body_end:\s*$", source,
    ):
        source = (
            source[:body_end.start()]
            + "rvgen_program_body_end:\n"
            + source[body_end.start():]
        )
    scratch_register, buffer_register = next(
        ((first, second) for first in range(5, 32) for second in range(5, 32)
         if first != second and frame_register not in {first, second}
         and kernel_register not in {first, second}),
        (None, None),
    )
    if scratch_register is None or buffer_register is None:
        raise ValueError("RISC-V-DV observer temporary registers unavailable")
    frame = lambda index: word * (index + 1)
    snapshot = [f"                  {store} x0, 16(x{buffer_register})"]
    for index in range(1, 32):
        output = 16 + index * 8
        if index == frame_register:
            snapshot.append(f"                  csrr x{scratch_register}, 0x340")
        else:
            snapshot.append(
                f"                  {load} x{scratch_register}, {frame(index)}(x{frame_register})"
            )
        snapshot.append(
            f"                  {store} x{scratch_register}, {output}(x{buffer_register})"
        )
        if xlen == 32:
            snapshot.append(f"                  sw x0, {output + 4}(x{buffer_register})")

    if xlen == 64:
        magic = [f"                  li x{scratch_register}, {_OBSERVATION_MAGIC}",
                 f"                  sd x{scratch_register}, 0(x{buffer_register})"]
    else:
        magic = [f"                  li x{scratch_register}, {_OBSERVATION_MAGIC & 0xffffffff}",
                 f"                  sw x{scratch_register}, 0(x{buffer_register})",
                 f"                  li x{scratch_register}, {_OBSERVATION_MAGIC >> 32}",
                 f"                  sw x{scratch_register}, 4(x{buffer_register})"]
    memory_registers = [
        register for register in range(5, 32)
        if register not in {
            frame_register, kernel_register, scratch_register, buffer_register,
        }
    ][:4]
    if len(memory_registers) != 4:
        raise ValueError("RISC-V-DV observer memory registers unavailable")
    memory_start, memory_end, memory_byte, memory_dest = memory_registers
    memory_snapshot = [
        f"                  la x{memory_start}, test_memory_start",
        f"                  la x{memory_end}, test_memory_end",
        f"                  sub x{memory_byte}, x{memory_end}, x{memory_start}",
        f"                  {store} x{memory_byte}, 272(x{buffer_register})",
        *([f"                  sw x0, 276(x{buffer_register})"] if xlen == 32 else []),
        f"                  addi x{memory_dest}, x{buffer_register}, {OBSERVATION_SIZE}",
        "rvobs_copy_loop:",
        f"                  beq x{memory_start}, x{memory_end}, rvobs_copy_done",
        f"                  lbu x{memory_byte}, 0(x{memory_start})",
        f"                  sb x{memory_byte}, 0(x{memory_dest})",
        f"                  addi x{memory_start}, x{memory_start}, 1",
        f"                  addi x{memory_dest}, x{memory_dest}, 1",
        "                  j rvobs_copy_loop",
        "rvobs_copy_done:",
    ]
    direct_frame = [
        f"                  {store} x{index}, {-512 + frame(index)}(x{stack_register})"
        for index in range(1, 32) if index != frame_register
    ]
    direct_test_done = [
        " .align 2".lstrip(),
        "rvobs_test_done:",
        *direct_frame,
        f"                  la x{scratch_register}, test_done",
        f"                  csrw 0x341, x{scratch_register}",
        f"                  li x{scratch_register}, 8",
        f"                  csrw 0x342, x{scratch_register}",
        f"                  addi x{frame_register}, x{stack_register}, -512",
        "                  j rvobs_from_trap",
    ]
    # Renode handles ebreak as a machine trap.  Its capsule setup places an
    # ebreak at 0x1010 that pauses the CPU; select that vector after the
    # mailbox is finalized so the DV exception handler cannot re-enter
    # test_done forever.  RVVM also stops directly on this ebreak, avoiding a
    # slow busy-loop in coverage builds.
    terminal = [
        f"                  li x{scratch_register}, 0x1010",
        f"                  csrw 0x305, x{scratch_register}",
        "                  ebreak",
        "                  j .",
    ]
    observer = "\n".join([
        ".section .text",
        ".align 2",
        *direct_test_done,
        "rvobs_from_trap:",
        f"                  la x{buffer_register}, obs_buf",
        f"                  csrr x{scratch_register}, 0x341",
        f"                  {store} x{scratch_register}, 8(x{buffer_register})",
        *snapshot,
        *magic,
        *memory_snapshot,
        f"                  li x{scratch_register}, 1",
        f"                  la x{buffer_register}, tohost",
        f"                  {store} x{scratch_register}, 0(x{buffer_register})",
        *terminal,
        ".section .data",
        ".align 3",
        ".globl obs_buf",
        "obs_buf:",
        f".zero {OBSERVATION_SIZE}",
        "",
    ])
    marker = re.search(r"(?m)^\.section\s+\.data\b[^\r\n]*(?:\r?\n|$)", source)
    if marker is None:
        return source + "\n" + observer
    data = source[marker.end():]
    stack_marker = re.search(r"(?m)^\.section\s+\.user_stack\b", data)
    data_body, data_tail = (
        (data[:stack_marker.start()], data[stack_marker.start():])
        if stack_marker is not None else (data, "")
    )
    memory_labels = tuple(
        re.search(rf"(?m)^\s*{name}:\s*$", data_body)
        for name in ("test_memory_start", "test_memory_end")
    )
    if all(memory_labels) and memory_labels[0].start() > memory_labels[1].start():
        raise ValueError("RISC-V-DV data memory range is reversed")
    if not all(memory_labels):
        data = "\n".join((
            ".globl test_memory_start",
            "test_memory_start:",
            data_body,
            ".globl test_memory_end",
            "test_memory_end:",
            data_tail,
        ))
    return _move_tohost_after_test_memory(source[:marker.start()] + observer + data)


def _custom_linux_user_harness(
    source: str, *, xlen: int = 64, initial_state: Mapping[str, object] | None = None,
    prologue_lines: tuple[str, ...] = (),
) -> str:
    """给自带 ``_start``/终止代码的 custom source 补最小观察出口。"""
    original_start = next((i for i, line in enumerate(source.splitlines()) if line.strip() == "_start:"), None)
    source_start_line = (original_start + 2) if original_start is not None else None
    source = _expand_asm_macros(source)
    lines = source.splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip() == "_start:"), None)
    if start is None:
        raise ValueError("custom source has no _start")
    section = re.compile(
        r"^\s*(?:\.section\s+([^,\s]+)|\.(bss|data|rodata|sdata|sbss)\b)"
    )
    split = next(
        (i for i, line in enumerate(lines[start + 1:], start + 1)
         if (match := section.match(line))
         and (match[1] or f".{match[2]}") != ".text"),
        len(lines),
    )
    text, tail = lines[:split], lines[split:]
    tail = [line for line in tail if not line.lower().lstrip().startswith(".section .note.gnu-stack")]
    blocks: list[list[str]] = []
    for line in tail:
        if section.match(line):
            blocks.append([])
        if blocks:
            blocks[-1].append(line)
    writable, readonly = [], []
    for block in blocks:
        match = section.match(block[0])
        name = (match[1] or f".{match[2]}") if match else ""
        if name in {".data", ".bss", ".sdata", ".sbss"}:
            block[0] = re.sub(r"^\s*.*$", ".section .data", block[0])
            writable.extend(block)
        else:
            readonly.extend(block)
    tail = writable + readonly
    source_start_line = source_start_line or start + 2
    text[start:start] = ["#line 10000"]
    start += 1
    for index, line in enumerate(text[:-1]):
        ebreak = re.match(
            r"^(\s*)(?:([0-9]+|[.$A-Za-z_][\w.$]*):\s*)?ebreak(?:\s*(?:#.*)?)?$",
            line,
        )
        if ebreak:
            label = f"{ebreak[2]}: " if ebreak[2] else ""
            text[index] = f"{ebreak[1]}{label}j write_tohost"
            continue
        inline = re.match(
            r"^(\s*)([0-9]+|[.$A-Za-z_][\w.$]*):\s+j\s+"
            r"\2[bf](\s*(?:#.*)?)$", line,
        )
        if inline:
            text[index] = f"{inline[1]}{inline[2]}: j write_tohost{inline[3]}"
            continue
        match = re.match(r"^(\s*)j\s+([0-9]+[bf]|[.$A-Za-z_][\w.$]*)(\s*(?:#.*)?)$", line)
        if match:
            target = match[2][:-1] if match[2][-1] in "bf" else match[2]
            previous = index - 1
            while previous >= 0 and (
                not text[previous].strip() or text[previous].lstrip().startswith("#line")
            ):
                previous -= 1
            label = re.match(r"^\s*([0-9]+|[.$A-Za-z_][\w.$]*):\s*$", text[previous]) \
                if previous >= 0 else None
            if label and label[1] == target:
                text[index] = f"{match[1]}j write_tohost{match[3]}"
                continue
        if not re.match(r"^\s*li\s+a7\s*,\s*93\b", line):
            continue
        next_index = index + 1
        while next_index < len(text) and text[next_index].lstrip().startswith("#line"):
            next_index += 1
        if next_index < len(text) and re.match(r"^\s*ecall\b", text[next_index]):
            text[next_index] = re.sub(r"^\s*ecall\b.*", "  j write_tohost", text[next_index])
    if not any(line.strip() == "init:" for line in text):
        text[start + 1:start + 1] = [
            "#line 10000", "init:", "  la sp, rvobs_user_stack_end", f"#line {source_start_line}",
        ]
    text.extend(["#line 10000", "test_done:", "write_tohost:", "  j exit_checkpoint"])
    synthetic_data = [".section .data"]
    if not re.search(r"(?m)^\s*tohost:", "\n".join(tail)):
        synthetic_data.append("tohost: .dword 0")
    synthetic_data.extend([
        ".align 3", "rvobs_user_stack:", ".zero 512", "rvobs_user_stack_end:",
    ])
    synthetic = "\n".join(text + synthetic_data + tail) + "\n"
    return linux_user_harness(
        synthetic, xlen=xlen, initial_state=initial_state,
        prologue_lines=prologue_lines,
    )


def _rax_harness(
    source: str, *, xlen: int = 64, initial_state: Mapping[str, object] | None = None,
    prologue_lines: tuple[str, ...] = (),
    base_harness=linux_user_harness,
) -> str:
    text = base_harness(
        source, xlen=xlen, initial_state=initial_state,
        prologue_lines=prologue_lines,
    )
    text, count = re.subn(
        r"(?ms)^\s*li a0, 1\n.*?^\s*ecall\n(?:\s*ebreak\n\s*wfi\n)?\s*li a0, 0\n\s*li a7, 93\n\s*ecall\n\s*j \.\s*$",
        "  li a0, 0\n  li a7, 93\n  ecall",
        text,
        count=1,
    )
    if count != 1:
        raise RuntimeError("RAX harness could not bind RVOBS1 exit transport")
    return text


def _rax_uart_transport(source: str) -> str:
    """通过 RAX 的 RISC-V 16550 MMIO 输出完整 RVOBS1 帧后退出。"""
    replacement = "\n".join((
        "rvobs_copy_done:",
        "  li t0, 0x10000000",
        "rax_uart_loop:",
        "  beqz a2, rax_uart_done",
        "  lbu t1, 0(a1)",
        "  sb t1, 0(t0)",
        "  addi a1, a1, 1",
        "  addi a2, a2, -1",
        "  j rax_uart_loop",
        "rax_uart_done:",
        "  li a0, 0",
        "  li a7, 93",
        "  ecall",
    ))
    source, count = re.subn(
        r"(?ms)^rvobs_copy_done:\s*\n.*?(?=^\.option pop\s*$)",
        replacement + "\n", source, count=1,
    )
    if count != 1:
        raise RuntimeError("RAX UART transport could not bind RVOBS1 exit")
    return source


def _localize_tohost(source: str) -> str:
    source = re.sub(r"(?m)^\s*\.(?:globl|global)\s+tohost\s*$\n?", "", source)
    source = re.sub(r"(?m)\.(?:globl|global)\s+tohost\s*;\s*", ".local tohost; ", source)
    if not re.search(r"(?m)(?:^|;)\s*\.local\s+tohost(?:\s*;|\s*$)", source):
        source = re.sub(r"(?m)(^|;)(\s*tohost\s*:)", r"\1.local tohost;\2", source, count=1)
    return source


def _move_tohost_after_test_memory(source: str) -> str:
    """Keep the terminal word outside the RVOBS1 payload destination."""
    lines = source.splitlines(keepends=True)
    memory_end = next(
        (index for index, line in enumerate(lines)
         if re.match(r"^\s*test_memory_end\s*:", line, re.IGNORECASE)),
        None,
    )
    if memory_end is None:
        return source
    definitions = [
        index for index, line in enumerate(lines[:memory_end])
        if re.search(r"(?<![A-Za-z0-9_.])tohost\s*:", line, re.IGNORECASE)
    ]
    if not definitions:
        return source
    if len(definitions) != 1:
        raise RuntimeError("tohost definition is ambiguous")
    definition = definitions[0]
    line = lines.pop(definition)
    memory_end -= 1
    lines.insert(memory_end + 1, line)
    return "".join(lines)


def _tohost_transport(source: str, *, xlen: int) -> str:
    store = "sd" if xlen == 64 else "sw"
    replacement = "\n".join((
        "rvobs_copy_done:",
        "  li a0, 1",
        "  lla a1, tohost",
        f"  {store} a0, 0(a1)",
        # Renode treats ebreak as a machine trap.  RISC-V-DV's startup frame
        # installs its own mtvec, whose generic exception path jumps back to
        # test_done; point the final sentinel at Renode's injected pause page
        # so a completed mailbox cannot turn into an observer loop.
        "  li t0, 0x1010",
        "  csrw mtvec, t0",
        "  ebreak",
        "  j .",
    ))
    source, count = re.subn(
        r"(?ms)^\s*rvobs_copy_done:\s*\n.*?^\s*j\s+\.\s*$",
        replacement, source, count=1,
    )
    if count != 1:
        raise RuntimeError("RVOBS1 tohost transport could not bind exit")
    if not re.search(r"(?m)(?:^|;)\s*tohost\s*:", source):
        source += "\n.section .data\n.align 3\ntohost: .dword 0\n"
    source = _localize_tohost(source)
    source = _move_tohost_after_test_memory(source)
    if not re.search(r"(?m)^\s*(?:\.globl\s+)?test_0\s*$", source):
        source = ".globl test_0\n.set test_0, _start\n" + source
    if not re.search(r"(?m)^\s*(?:\.globl\s+)?test_entry\s*(?::|$)", source):
        source = ".globl test_entry\n.set test_entry, _start\n" + source
    return _fast_copy_loop(source, xlen)


def _add_torture_observer(source_path: Path, header_path: Path, *, xlen: int = 64) -> dict[str, object]:
    """给 torture 裸机退出宏绑定同一份 RVOBS1 guest-memory 帧。"""
    if xlen not in {32, 64}:
        raise ValueError("xlen must be 32 or 64")
    source = source_path.read_text(encoding="utf-8")
    header = header_path.read_text(encoding="utf-8")
    exit_sequence = "  la t0, tohost; sd TESTNUM, 0(t0); ebreak"
    if xlen == 32:
        exit_sequence = "  la t0, tohost; sw TESTNUM, 0(t0); ebreak"
    hits = header.count(exit_sequence)
    # The normalized header has PASS, FAIL, and an exception-tohost exit.
    # Bind every site so no guest path can skip the observation frame.
    if hits < 2:
        raise RuntimeError(f"torture observer exit binding expected at least 2 header sites, got {hits}")
    if source.count("RVTEST_CODE_BEGIN") != 1 or source.count("RVTEST_CODE_END") != 1:
        raise RuntimeError("torture observer could not bind code section")
    header_path.write_text(
        header.replace(
            exit_sequence,
            "  la sp, rvobs_stack_top; j rvobs_exit_checkpoint",
        ),
        encoding="utf-8",
    )
    store = "sd" if xlen == 64 else "sw"
    gpr_lines: list[str] = []
    for index in range(32):
        if index in {5, 6}:
            continue
        offset = 16 + index * 8
        gpr_lines.append(f"\t{store} x{index}, {offset}(t0)")
        if xlen == 32:
            gpr_lines.append(f"\tsw zero, {offset + 4}(t0)")
    observer = "\n".join([
        "",
        ".section .text",
        ".option push",
        ".option norelax",
        ".option norvc",
        ".align 2",
        ".globl rvobs_exit_checkpoint",
        "rvobs_exit_checkpoint:",
        "\tj rvobs_emit",
        "rvobs_emit:",
        "\t" + store + " t0, -8(sp)",
        "\t" + store + " t1, -16(sp)",
        "\tla t0, obs_buf",
        *gpr_lines,
        "\tld t1, -8(sp)" if xlen == 64 else "\tlw t1, -8(sp)",
        "\t" + store + " t1, 56(t0)",
        "\tld t1, -16(sp)" if xlen == 64 else "\tlw t1, -16(sp)",
        "\t" + store + " t1, 64(t0)",
        "\tla t1, rvobs_exit_checkpoint",
        "\t" + store + " t1, 8(t0)",
        *( ["\tsw zero, 12(t0)"] if xlen == 32 else [] ),
        "\tla t1, test_memory_start",
        "\tla t2, test_memory_end",
        "\tsub t3, t2, t1",
        "\t" + store + " t3, 272(t0)",
        *( ["\tsw zero, 276(t0)"] if xlen == 32 else [] ),
        "\taddi t4, t0, 280",
        "rvobs_copy_loop:",
        "\tbeq t1, t2, rvobs_copy_done",
        "\tlbu t3, 0(t1)",
        "\tsb t3, 0(t4)",
        "\taddi t1, t1, 1",
        "\taddi t4, t4, 1",
        "\tj rvobs_copy_loop",
        "rvobs_copy_done:",
        "rvobs_tohost:",
        "\tla t1, tohost",
        # The complete architectural snapshot already contains TESTNUM.  The
        # terminal value is an adapter transport status: RAX emits RVOBS1
        # only after the protocol pass value, so normalize it to one.
        "\tli t2, 1",
        "\t" + store + " t2, 0(t1)",
        "\tebreak",
        "\tj .",
        ".option pop",
        "",
        ".section .data",
        ".align 3",
        ".globl obs_buf",
        "obs_buf:",
        ".byte 0x52, 0x56, 0x4f, 0x42, 0x53, 0x31, 0, 0",
        ".zero 272",
        ".globl test_memory_start",
        "test_memory_start = test_memory",
        ".globl test_memory_end",
        "test_memory_end = loop_count",
        ".align 3",
        "rvobs_stack:",
        ".zero 512",
        "rvobs_stack_top:",
        "",
    ])
    marker = "RVTEST_CODE_END\n"
    if source.count(marker) != 1:
        raise RuntimeError("riscv-dv source must contain exactly one RVTEST_CODE_END marker")
    source = source.replace(marker, marker + observer, 1)
    source_path.write_text(source, encoding="utf-8")
    return {
        "status": "passed",
        "exit_sites": hits,
        "xlen": xlen,
        "observation_size": OBSERVATION_SIZE,
        "memory_symbols": ["test_memory_start", "test_memory_end"],
    }



def build_linux_user(
    source_path: str | Path,
    output_path: str | Path,
    run_params: object,
) -> Path:
    """在 Linux execution plane 构建带 RVOBS1 的临时 harness ELF。"""
    if os.name == "nt":
        raise RuntimeError("Linux-user harness must run in the Linux execution plane")
    params = {} if run_params is None else run_params
    if not isinstance(params, Mapping):
        raise ValueError("run_params must be an object")
    try:
        build_timeout = float(params.get("timeout", 120))
    except (TypeError, ValueError) as error:
        raise ValueError("build timeout must be positive and finite") from error
    if not math.isfinite(build_timeout) or build_timeout <= 0:
        raise ValueError("build timeout must be positive and finite")
    initial_state = params.get("initial_state")
    if initial_state is not None and not isinstance(initial_state, Mapping):
        raise ValueError("initial_state must be an object")
    compiler = str(params.get("compiler") or "riscv64-linux-gnu-gcc")
    gcc_opts = shlex.split(str(params.get("gcc_opts") or ""))
    profile = str(params.get("isa") or params.get("isa_profile") or "rv64imc").strip()
    march, _, _ = _toolchain_profile(profile)
    extensions = enabled_extensions(profile)
    if "e" in extensions:
        raise RuntimeError("RV32E Linux-user harness is unsupported")
    abi = str(params.get("mabi") or ("ilp32" if profile.lower().startswith("rv32") else "lp64")).strip()
    if not params.get("mabi"):
        abi += "d" if {"q", "d"} & extensions else "f" if "f" in extensions else ""
    elif not mabi_matches_isa(profile, abi):
        raise RuntimeError("explicit MABI does not match ISA profile")
    source_path = source_path if isinstance(source_path, Path) else Path(source_path)
    output_path = output_path if isinstance(output_path, Path) else Path(output_path)
    output_path = output_path.resolve()
    if source_path.resolve() == output_path:
        raise ValueError("input and output paths must differ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    harness_path = output_path.with_suffix(".S")
    if harness_path.resolve() == source_path.resolve():
        # ProgramVariant builds place their source beside the ELF.  Keep the
        # generated harness basename stable across baseline and pair rebuilds;
        # otherwise GCC embeds ``program.harness.S`` in DWARF and the ELF hash
        # changes even when .text/.data are byte-identical.
        harness_path = output_path.parent / ".rq1-harness" / harness_path.name
    harness_path.parent.mkdir(parents=True, exist_ok=True)
    xlen = 32 if profile.lower().startswith("rv32") else 64
    source = source_path.read_text(encoding="utf-8")
    has_dv_frame = bool(re.search(r"(?m)^\s*(?:init|h0_start):\s*$", source)) and all(
        re.search(rf"(?m)^\s*{name}:\s*$", source)
        for name in ("test_done", "write_tohost")
    )
    if isinstance(initial_state, Mapping) and initial_state.get("privilege") not in (
        None, "u", "U", "user", "U-mode",
    ):
        privileged_source = re.search(
            r"(?mi)^\s*(?:mret|sret|wfi|sfence|hfence)(?=\s|$)"
            r"|^\s*csr[a-z.]*\s+(?:[^,\r\n]*,\s*)?"
            r"(?:0x[0-9a-f]+|m[a-z][a-z0-9]*|s(?![0-9p])[a-z][a-z0-9]*|h[a-z][a-z0-9]*)\b",
            source,
        )
        if params.get("harness") == "riscv-dv" or has_dv_frame or privileged_source is None:
            # ponytail: 完整 DV 帧的特权序言属于 harness；无帧自定义源仍保留升级边界。
            initial_state = {
                key: value for key, value in initial_state.items() if key != "privilege"
            }
        elif not params.get("allow_privileged_initial_state"):
            raise RuntimeError("initial-state-privilege-not-materialized")
        else:
            # target capsule 由适配器以其声明的初始特权级启动；Linux-user
            # reference 只能保留可构建的寄存器初始化，特权级差异必须留在证据中。
            initial_state = {
                key: value for key, value in initial_state.items() if key != "privilege"
            }
    include_pattern = re.compile(
        r'(?m)^[ \t]*(?:\.include|#\s*include)[ \t]+"(user_init\.s|user_define\.h)"'
        r'(?P<tail>[^\r\n]*)(?:\r?\n|$)'
    )
    original_source, contents = source, {}
    target_name = str(params.get("target") or profile.split("_", 1)[0])
    for name in ("user_init.s", "user_define.h"):
        if not re.search(rf'(?m)^[ \t]*(?:\.include|#\s*include)[ \t]+"{re.escape(name)}"[^\r\n]*(?:\r?\n|$)', source):
            continue
        candidates = (
            source_path.parent / name,
            source_path.parent.parent / "user_extension" / name,
            source_path.parent.parent / name,
            source_path.parent.parent.parent / "user_extension" / name,
            source_path.parent.parent.parent / name,
            source_path.parent.parent.parent / "target" / target_name / name,
        )
        dependency = next((path for path in candidates if path.is_file()), None)
        if dependency is None:
            raise RuntimeError(f"RISC-V-DV source requires missing {name}")
        if dependency.resolve() == output_path:
            raise ValueError("input and output paths must differ")
        contents[name] = dependency.read_text(encoding="utf-8")
    if any(path.exists() or path.is_symlink() for path in (harness_path, output_path)):
        raise ValueError("build-artifact already exists")
    prologue_lines = tuple(params.get("sail_prologue_lines") or ())
    if any(not isinstance(line, str) or not line for line in prologue_lines):
        raise ValueError("sail_prologue_lines must contain non-empty strings")
    text_address = str(params.get("text_address") or "0x80000000")
    def expand_include(match: re.Match[str]) -> str:
        name, line = match[1], original_source.count("\n", 0, match.start()) + 1
        content = contents.get(name)
        if content is None:
            return match[0]
        tail = match.group("tail")
        return f"#line {line}\n{content.rstrip(chr(10) + chr(13))}\n#line {line + 1}\n" + (
            f"{tail}\n" if tail else ""
        )
    source = include_pattern.sub(expand_include, source)
    bare_metal = params.get("bare_metal") is True
    if params.get("harness") == "riscv-dv" and not bare_metal:
        source = _neutralize_rvdv_user_instructions(source)
    custom = params.get("harness") == "custom"
    has_frame = any(
        re.search(rf"(?m)^\s*{name}:\s*$", source) for name in ("init", "h0_start")
    ) and all(
        re.search(rf"(?m)^\s*{name}:\s*$", source)
        for name in ("test_done", "write_tohost")
    )
    transport = str(params.get("transport") or "linux")
    if transport not in {"linux", "tohost", "rax-uart"}:
        raise ValueError("unsupported observer transport")
    if bare_metal and (params.get("harness") == "riscv-dv" or has_dv_frame):
        harness_source = _bare_metal_rvdv_harness(
            source, xlen=xlen,
        )
    else:
        base_harness = linux_user_harness if not custom or has_frame else _custom_linux_user_harness
        # The RAX syscall cleanup is only needed for its Linux-user transport.
        # The tohost and UART transports must retain ``rvobs_copy_done`` so they
        # can replace that block with their native terminal channel.
        if params.get("target_backend") == "rax-riscv64" and transport not in {"tohost", "rax-uart"}:
            harness = lambda source, **kwargs: _rax_harness(
                source, base_harness=base_harness, **kwargs,
            )
        else:
            harness = base_harness
        harness_source = harness(
            source, xlen=xlen, initial_state=initial_state,
            prologue_lines=prologue_lines,
        )
        if transport == "tohost":
            harness_source = _tohost_transport(harness_source, xlen=xlen)
        elif transport == "rax-uart":
            harness_source = _rax_uart_transport(harness_source)
    harness_path.write_text(harness_source, encoding="utf-8")
    linker = params.get("linker")
    text_segment = params.get("text_segment")
    global_pointer = str(params.get("global_pointer") or hex(int(text_address, 0) + 0x1000))
    command = [
        compiler,
        "-g",
        "-save-temps=obj",
        f"-ffile-prefix-map={harness_path.parent}=.",
        f"-fdebug-prefix-map={harness_path.parent}=.",
        f"-march={march}",
        f"-mabi={abi}",
        "-fno-pic",
        *gcc_opts,
        "-nostdlib",
        "-nostartfiles",
        "-static",
        "-no-pie",
        "-I",
        str(source_path.parent),
        "-Wl,--build-id=none",
        "-Wl,--no-relax",
        f"-Wl,--defsym=__global_pointer$={global_pointer}",
        *(["-Wl,-T", str(linker)] if linker else [f"-Wl,-Ttext-segment={text_segment}" if text_segment else f"-Wl,-Ttext={text_address}"]),
        harness_path.name,
        "-o",
        str(output_path),
    ]
    try:
        result = subprocess.run(
            command, cwd=harness_path.parent, capture_output=True, text=True,
            check=False, timeout=build_timeout,
        )
    except subprocess.TimeoutExpired as exc:
        harness_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        raise RuntimeError("Linux-user harness build timed out") from exc
    except OSError as exc:
        harness_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"Linux-user harness build failed: {exc}") from exc
    if result.returncode != 0:
        harness_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"Linux-user harness build failed: {result.stderr.strip()}")
    return output_path


def build_bare_metal_capsule(
    record: Mapping[str, object], output_path: str | Path,
) -> Path:
    """把一个直接测试用例物化为四个 capsule 后端共用的 RVOBS1 ELF。"""
    testcase = record["testcase"]
    profile = str(record.get("profile") or "rv64i")
    extensions = enabled_extensions(profile)
    rv32 = profile.lower().startswith("rv32")
    store, load = ("sw", "lw") if rv32 else ("sd", "ld")
    abi = ("ilp32" if rv32 else "lp64") + (
        "d" if "d" in extensions else "f" if "f" in extensions else ""
    )
    code = bytes.fromhex(str(testcase["code_hex"]))
    if len(code) % 2:
        raise ValueError("capsule code must be instruction-aligned")
    memory = next(
        item for item in testcase.get("initial_memory_regions", [])
        if item.get("region_id") == "test-memory"
    )
    memory_bytes = bytes.fromhex(str(memory["data_hex"]))
    initial = testcase.get("initial_gpr") or [0] * 32
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    source = output.with_suffix(".S")
    linker = output.with_suffix(".ld")
    words = "\n".join(
        f"  .2byte 0x{int.from_bytes(code[offset:offset + 2], 'little'):04x}"
        for offset in range(0, len(code), 2)
    )
    setup = "\n".join(
        f"  li x{index}, {int(value)}"
        for index, value in enumerate(initial[1:], 1)
        if index not in {2, 14, 29, 30, 31} and int(value) != 0
    )
    data = ", ".join(f"0x{value:02x}" for value in memory_bytes) or "0"
    def store_u64(register: int, offset: int) -> list[str]:
        return [f"  {store} x{register}, {offset}(x31)"] + (
            [f"  sw zero, {offset + 4}(x31)"] if rv32 else []
        )

    snapshots = "\n".join(
        line for register in range(29) for line in store_u64(register, 16 + register * 8)
    )
    snapshot_x29 = "\n".join(store_u64(29, 248))
    checkpoint = "\n".join(store_u64(30, 8))
    restore_x30 = "\n".join(store_u64(28, 256))
    restore_x31 = "\n".join(store_u64(28, 264))
    memory_size = "\n".join(store_u64(30, 272))
    magic = "\n".join(
        (
            f"  li x30, {_OBSERVATION_MAGIC & 0xffffffff}",
            "  sw x30, 0(x31)",
            f"  li x30, {_OBSERVATION_MAGIC >> 32}",
            "  sw x30, 4(x31)",
        ) if rv32 else (
            f"  li x30, {_OBSERVATION_MAGIC}",
            "  sd x30, 0(x31)",
        )
    )
    source.write_text(
        f""".option norvc
.section .text
.globl _start
_start:
  la x2, stack_top
  mv x29, x2
{setup}
  la x14, test_memory
entry_checkpoint:
{words}
exit_checkpoint:
  {store} x30, {-8 if rv32 else -16}(x29)
  {store} x31, {-4 if rv32 else -8}(x29)
  la x31, result_buffer
  la x30, exit_checkpoint
{checkpoint}
{snapshots}
{snapshot_x29}
  {load} x28, {-8 if rv32 else -16}(x29)
{restore_x30}
  {load} x28, {-4 if rv32 else -8}(x29)
{restore_x31}
  li x30, {len(memory_bytes)}
{memory_size}
  la x30, test_memory
  addi x28, x31, {OBSERVATION_HEADER_SIZE}
  li x27, {len(memory_bytes)}
  beq x27, x0, capsule_observe
capsule_copy:
  lbu x26, 0(x30)
  sb x26, 0(x28)
  addi x30, x30, 1
  addi x28, x28, 1
  addi x27, x27, -1
  bne x27, x0, capsule_copy
capsule_observe:
{magic}
  # The mailbox is the common bare-metal observation channel.  The adapters
  # read it from guest memory; the explicit tohost write is the terminal
  # outcome channel.  Do not route this frame through an optional UART: a
  # missing MMIO device would trap before tohost on otherwise valid targets.
  li x30, 1
  la x28, tohost
  {store} x30, 0(x28)
  ebreak
.Lcapsule_halt:
  j .Lcapsule_halt
.section .data
.align 3
result_buffer:
.zero {OBSERVATION_HEADER_SIZE}
test_memory:
.byte {data}
result_end:
.section .tohost,"aw",@progbits
.align 3
.local tohost
tohost:
.dword 0
.section .bss
.align 12
stack_area:
.zero 4096
stack_top:
""",
        encoding="utf-8",
    )
    linker.write_text(
        "SECTIONS { . = 0x80000000; .text : { *(.text*) } "
        ". = 0x81000000; .data : { *(.data*) } "
        ".tohost : { *(.tohost*) } . = ALIGN(0x1000); "
        ".bss : { *(.bss*) *(COMMON) } }\n",
        encoding="utf-8",
    )
    command = [
        "riscv64-linux-gnu-gcc", f"-march={profile}", f"-mabi={abi}", "-nostdlib",
        "-nostartfiles", "-static", "-no-pie", "-Wl,--build-id=none", "-Wl,--no-relax",
        f"-Wl,-T,{linker}", str(source), "-o", str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=120)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "bare-metal capsule build failed")
    return output


__all__ = [
    "OBSERVATION_SIZE",
    "build_bare_metal_capsule",
    "build_linux_user",
    "linux_user_harness",
]
