#!/usr/bin/env python3
"""RISC-V-DV 兼容启动器，补齐旧 pygen 的最小运行时契约。"""

from __future__ import annotations

import atexit
import os
from pathlib import Path
import random
import re
import runpy
import shutil
import sys
import tempfile


OLD = """    def gen_callstack(self, main_program, sub_program,
                      sub_program_name, num_sub_program):
        if num_sub_program != 0:
            callstack_gen = riscv_callstack_gen()
            callstack_gen.init(num_sub_program + 1)
            if callstack_gen.randomize():
                idx = 0
                # Insert the jump instruction based on the call stack
                for i in range(len(callstack_gen.program_h)):
                    for j in range(len(callstack_gen.program_h.sub_program_id)):
                        idx += 1
                        pid = callstack_gen.program_id[i].sub_program_id[j] - 1
                        logging.info("Gen jump instr %0s -> sub[%0d] %0d", i, j, pid + 1)
                        if(i == 0):
                            self.main_program[i].insert_jump_instr(sub_program_name[pid], idx)
                        else:
                            self.sub_program[i - 1].insert_jump_instr(sub_program_name[pid], idx)
            else:
                logging.critical("Failed to generate callstack")
                sys.exit(1)
        logging.info("Randomizing call stack..done")
"""
CALLSTACK = """import random


class _Program:
    def __init__(self, sub_program_id=()):
        self.sub_program_id = list(sub_program_id)


class riscv_callstack_gen:
    def init(self, program_cnt):
        self.program_cnt = int(program_cnt)
        self.program_h = [_Program()] + [
            _Program() for _ in range(1, self.program_cnt)
        ]

    def randomize(self):
        if self.program_cnt <= 1:
            return True
        # Build a randomized acyclic call tree. Every sub-program gets a caller,
        # and each parent calls its children before its random instruction body.
        levels = [0]
        for _ in range(1, self.program_cnt):
            previous = levels[-1]
            levels.append(random.randint(max(1, previous), previous + 1))
        for level in range(max(levels)):
            parents = [i for i, item in enumerate(levels) if item == level]
            children = [i for i, item in enumerate(levels) if item == level + 1]
            random.shuffle(children)
            for child in children:
                self.program_h[random.choice(parents)].sub_program_id.append(child)
        return True
"""

NEW = """    def gen_callstack(self, main_program, sub_program,
                      sub_program_name, num_sub_program):
        if num_sub_program != 0:
            callstack_gen = riscv_callstack_gen()
            callstack_gen.init(num_sub_program + 1)
            if not callstack_gen.randomize():
                raise RuntimeError("RISC-V-DV call-stack randomization failed")
            for program_index, program in enumerate(callstack_gen.program_h):
                # The main sequence is passed per hart.  The pinned generator
                # rebuilds self.sub_program for that hart, so its list is the
                # authoritative set of sub-program sequences.
                target = main_program if program_index == 0 else self.sub_program[program_index - 1]
                for jump_index, sub_program_id in enumerate(program.sub_program_id, 1):
                    index = int(sub_program_id) - 1
                    if not 0 <= index < len(sub_program_name):
                        raise RuntimeError("RISC-V-DV call-stack target is out of range")
                    target.insert_jump_instr(sub_program_name[index], jump_index)
        logging.info("Randomizing call stack..done")
"""

SEQUENCE_JUMP = """    def insert_jump_instr(self, target_label, idx):
        from pygen_src.isa.riscv_instr import riscv_instr

        call = riscv_instr.get_instr(riscv_instr_name_t.JAL)
        with call.randomize_with():
            call.rd == cfg.ra
        call.rd = cfg.ra
        call.imm_str = target_label
        call.idx = idx
        call.has_label = 0
        call.atomic = 1
        call.comment = "call {}".format(target_label)
        # Place calls after an atomic stack prologue when one exists. Calls at
        # function entry make every node in the generated acyclic graph reachable.
        call_sites = [i for i, instr in enumerate(self.instr_stream.instr_list)
                      if not instr.atomic]
        if not call_sites:
            raise RuntimeError("RISC-V-DV program has no call insertion point")
        self.instr_stream.insert_instr_stream([call], min(call_sites))
"""


_INIT_GPR_LOOP_CONTEXT = (
    "        for i in range(rcs.NUM_GPR):\n"
    "            if i in [cfg.sp.value, cfg.tp.value]:\n"
)
_INIT_GPR_LOOP_PATCH = (
    "        # LRSV-INT's pinned interpreter writes rd=x0 instead of preserving\n"
    "        # the architectural zero register; do not emit a fake x0 init.\n"
    "        for i in range(rcs.NUM_GPR):\n"
    "            if i in [0, cfg.sp.value, cfg.tp.value]:\n"
)

_RESERVED_REG_CONTEXT = (
    "        self.reserved_regs = vsc.list_t(vsc.enum_t(riscv_reg_t))\n"
)
_RESERVED_REG_PATCH = (
    "        self.reserved_regs = vsc.list_t(vsc.enum_t(riscv_reg_t))\n"
    "        # Keep random rd choices away from x0 for the pinned LRSV-INT\n"
    "        # interpreter; reads of x0 remain legal and continue to mean zero.\n"
    "        self.reserved_regs.append(riscv_reg_t.ZERO)\n"
)
_POST_RANDOMIZE_CONTEXT = (
    "        self.reserved_regs.extend((self.tp, self.sp, self.scratch_reg))\n"
)
_POST_RANDOMIZE_PATCH = (
    "        # The compatibility call-stack uses cfg.ra as its link register;\n"
    "        # random bodies must not overwrite it before the return routine.\n"
    "        self.reserved_regs.extend((self.tp, self.sp, self.scratch_reg, self.ra))\n"
)

_WORKER_SEED_CONTEXT = (
    "        else:\n"
    "            # Generate random seed value everytime for multiple test iterations\n"
    "            rand_seed = random.getrandbits(31)\n"
)
_WORKER_SEED_PATCH = (
    "        else:\n"
    "            # Keep every worker reproducible within the generated batch.\n"
    "            rand_seed = int(cfg.argv.seed.split(\"--\")[0]) + self.start_idx + num\n"
)

_PYFLOW_COMMAND_CONTEXT = (
    "      python3 <cwd>/pygen/pygen_src/test/<test_name>.py <sim_opts>"
)

_STACK_SEQUENCE_CONTEXT = """        # TODO Commenting for now as it is blocking sub_program
        # if not is_main_program:
        #     self.gen_stack_enter_instr()
        #     self.gen_stack_exit_instr()
        logging.info("Finishing instruction generation")
"""
_STACK_SEQUENCE_PATCH = """        body = self.instr_stream.instr_list
        if not is_main_program:
            self.instr_stream.instr_list = []
            self.gen_stack_enter_instr()
            self.instr_stream.instr_list.extend(body)
            self.gen_stack_exit_instr()
        logging.info("Finishing instruction generation")
"""
_BRANCH_TARGET_CONTEXT = """        for instr in self.directed_instr:
            self.instr_stream.insert_instr_stream(instr.instr_list)
        # Assign an index for all instructions, these indexes wont change
"""
_BRANCH_TARGET_PATCH = """        for instr in self.directed_instr:
            self.instr_stream.insert_instr_stream(instr.instr_list)
        stack_exit = self.instr_stack_exit.instr_list
        if stack_exit:
            exit_ids = {id(instr) for instr in stack_exit}
            stream = self.instr_stream.instr_list
            if sum(id(instr) in exit_ids for instr in stream) != len(stack_exit):
                raise RuntimeError("RISC-V-DV stack epilogue is missing from its program")
            self.instr_stream.instr_list = [
                instr for instr in stream if id(instr) not in exit_ids
            ]
        if not cfg.no_branch_jump:
            from pygen_src.isa.riscv_instr import riscv_instr
            terminal = riscv_instr.get_instr(riscv_instr_name_t.NOP)
            terminal.atomic = 0
            terminal.has_label = 1
            self.instr_stream.instr_list.append(terminal)
        if stack_exit:
            self.instr_stream.instr_list.extend(stack_exit)
        # Assign an index for all instructions, these indexes wont change
"""

_BRANCH_POST_PROCESS_CONTEXT = """        while j < len(self.instr_stream.instr_list):
            if((self.instr_stream.instr_list[j].category == riscv_instr_category_t.BRANCH) and
                    (not self.instr_stream.instr_list[j].branch_assigned) and
                    (not self.instr_stream.instr_list[j].is_illegal_instr)):
                # Post process the branch instructions to give a valid local label
                # Here we only allow forward branch to avoid unexpected infinite loop
                # The loop structure will be inserted with a separate routine using
                # reserved loop registers
                branch_target_label = 0
                branch_target_label = self.instr_stream.instr_list[j].idx + \\
                    branch_idx[branch_cnt]
                if(branch_target_label >= label_idx):
                    branch_target_label = label_idx - 1
                branch_cnt += 1
                if(branch_cnt == len(branch_idx)):
                    branch_cnt = 0
                    random.shuffle(branch_idx)
                logging.info("Processing branch instruction[%0d]:%0s # %0d -> %0d", j,
                             self.instr_stream.instr_list[j].convert2asm(),
                             self.instr_stream.instr_list[j].idx, branch_target_label)
                self.instr_stream.instr_list[j].imm_str = "{}f".format(branch_target_label)
                self.instr_stream.instr_list[j].branch_assigned = 1
                branch_target[branch_target_label] = 1
            # Remove the local label which is not used as branch target
            if(self.instr_stream.instr_list[j].has_label and
                    self.instr_stream.instr_list[j].is_local_numeric_label):
                idx = int(self.instr_stream.instr_list[j].label)
                if not branch_target[idx]:
                    self.instr_stream.instr_list[j].has_label = 0
            j += 1
        logging.info("Finished post-processing instructions")
"""

_BRANCH_POST_PROCESS_PATCH = """        # 先收集完整序列中的分支目标；否则末尾分支回退到旧标签时，
        # 目标标签可能已经在前向扫描中被删掉，留下未定义的 ``Nf``。
        while j < len(self.instr_stream.instr_list):
            if((self.instr_stream.instr_list[j].category == riscv_instr_category_t.BRANCH) and
                    (not self.instr_stream.instr_list[j].branch_assigned) and
                    (not self.instr_stream.instr_list[j].is_illegal_instr)):
                # Post process the branch instructions to give a valid local label
                # Here we only allow forward branch to avoid unexpected infinite loop
                # The loop structure will be inserted with a separate routine using
                # reserved loop registers
                branch_target_label = self.instr_stream.instr_list[j].idx + \\
                    branch_idx[branch_cnt]
                if(branch_target_label >= label_idx):
                    branch_target_label = label_idx - 1
                branch_cnt += 1
                if(branch_cnt == len(branch_idx)):
                    branch_cnt = 0
                    random.shuffle(branch_idx)
                logging.info("Processing branch instruction[%0d]:%0s # %0d -> %0d", j,
                             self.instr_stream.instr_list[j].convert2asm(),
                             self.instr_stream.instr_list[j].idx, branch_target_label)
                self.instr_stream.instr_list[j].imm_str = "{}f".format(branch_target_label)
                self.instr_stream.instr_list[j].branch_assigned = 1
                branch_target[branch_target_label] = 1
            j += 1
        # 分支目标已全部知道后再删除未使用的局部数字标签。
        for instr in self.instr_stream.instr_list:
            if instr.has_label and instr.is_local_numeric_label:
                idx = int(instr.label)
                if not branch_target[idx]:
                    instr.has_label = 0
        logging.info("Finished post-processing instructions")
"""


_LOCAL_LABEL = re.compile(r"^\s*(?P<label>[0-9]+):")
_SUBPROGRAM_LABEL = re.compile(r"^\s*sub_[0-9]+:\s*", re.IGNORECASE)
_LOCAL_FORWARD_REFERENCE = re.compile(r"(?<![A-Za-z0-9_.])(?P<label>[0-9]+)f\b")
_LOCAL_BRANCH_LINE = re.compile(
    r"^(?P<indent>\s*)(?:(?P<label>[0-9]+):\s*)?"
    r"(?P<mnemonic>(?:c\.)?b[A-Za-z0-9.]*)\b(?P<rest>.*?)(?P<newline>\r?\n)?$",
    re.IGNORECASE,
)


def _repair_local_branch_references(output: Path) -> None:
    """Repair numeric branch direction after the pinned generator reorders labels.

    The old pygen emits ``Nf`` even when its branch target was moved before the
    instruction by stack/call insertion.  GNU as quite correctly rejects such
    a reference.  Preserve valid forward references within one generated
    program, but turn a reordered backward branch or a reference crossing a
    ``sub_N`` boundary into a same-width no-op.  The generator contract says
    these branches are local and forward-only; a cross-program target skips a
    stack epilogue and corrupts the call-stack link before the checkpoint.
    """
    for path in sorted(output.rglob("*")):
        if path.suffix not in {".S", ".s"} or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines(keepends=True)
        labels: dict[str, list[int]] = {}
        for index, line in enumerate(lines):
            match = _LOCAL_LABEL.match(line)
            if match:
                labels.setdefault(match.group("label"), []).append(index)
        subprogram_starts = [
            index for index, line in enumerate(lines)
            if _SUBPROGRAM_LABEL.match(line)
        ]
        changed = False
        repaired: list[str] = []
        for index, line in enumerate(lines):
            backward = False
            cross_subprogram = False
            next_subprogram = next(
                (position for position in subprogram_starts if position > index),
                None,
            )

            def replace(match: re.Match[str]) -> str:
                nonlocal backward, cross_subprogram, changed
                label = match.group("label")
                positions = labels.get(label, [])
                forward = [position for position in positions if position > index]
                if forward:
                    if next_subprogram is not None and forward[0] >= next_subprogram:
                        cross_subprogram = True
                    return match.group(0)
                if any(position < index for position in positions):
                    backward = True
                return match.group(0)

            repaired_line = _LOCAL_FORWARD_REFERENCE.sub(replace, line)
            if backward or cross_subprogram:
                branch = _LOCAL_BRANCH_LINE.match(line)
                if branch is not None:
                    comment = ""
                    if "#" in branch.group("rest"):
                        comment = " #" + branch.group("rest").split("#", 1)[1]
                    replacement = "c.nop" if branch.group("mnemonic").lower().startswith("c.") else "nop"
                    label = f"{branch.group('label')}: " if branch.group("label") else ""
                    repaired_line = (
                        f"{branch.group('indent')}{label}{replacement}{comment}"
                        f"{branch.group('newline') or ''}"
                    )
                changed = True
            repaired.append(repaired_line)
        if changed:
            path.write_text("".join(repaired), encoding="utf-8")


def _patch_generator_init(source: str) -> str:
    if source.count(_INIT_GPR_LOOP_CONTEXT) != 1:
        raise ValueError("RISC-V-DV x0 initialization patch context changed")
    # cfg.tp is a reserved kernel-stack register.  It is initialized by
    # pre_enter_privileged_mode() and must survive init_gpr(); writing zero
    # here makes every machine trap save its frame at 0xffff... and loop.
    return source.replace(_INIT_GPR_LOOP_CONTEXT, _INIT_GPR_LOOP_PATCH, 1)


def _patch_reserved_regs(source: str) -> str:
    if source.count(_RESERVED_REG_CONTEXT) != 1:
        raise ValueError("RISC-V-DV reserved-register patch context changed")
    source = source.replace(_RESERVED_REG_CONTEXT, _RESERVED_REG_PATCH, 1)
    if source.count(_POST_RANDOMIZE_CONTEXT) != 1:
        raise ValueError("RISC-V-DV post-randomize patch context changed")
    return source.replace(_POST_RANDOMIZE_CONTEXT, _POST_RANDOMIZE_PATCH, 1)


def _patch_worker_seed(source: str) -> str:
    if source.count(_WORKER_SEED_CONTEXT) != 1:
        raise ValueError("RISC-V-DV worker-seed patch context changed")
    return source.replace(_WORKER_SEED_CONTEXT, _WORKER_SEED_PATCH, 1)


def _patch_pyflow_command(source: str, test_dir: Path) -> str:
    if source.count(_PYFLOW_COMMAND_CONTEXT) != 1:
        raise ValueError("RISC-V-DV pyflow command patch context changed")
    command = f"      python3 {test_dir}/<test_name>.py <sim_opts>"
    return source.replace(_PYFLOW_COMMAND_CONTEXT, command, 1)


def _patch_generator_sequence(source: str) -> str:
    if source.count(_STACK_SEQUENCE_CONTEXT) != 1:
        raise ValueError("RISC-V-DV sub-program stack patch context changed")
    source = source.replace(_STACK_SEQUENCE_CONTEXT, _STACK_SEQUENCE_PATCH, 1)
    if source.count(_BRANCH_TARGET_CONTEXT) != 1:
        raise ValueError("RISC-V-DV branch-target patch context changed")
    source = source.replace(_BRANCH_TARGET_CONTEXT, _BRANCH_TARGET_PATCH, 1)
    if source.count(_BRANCH_POST_PROCESS_CONTEXT) != 1:
        raise ValueError("RISC-V-DV branch post-process patch context changed")
    return source.replace(_BRANCH_POST_PROCESS_CONTEXT, _BRANCH_POST_PROCESS_PATCH, 1)


def _patch_riscv_instr(source: str) -> str:
    replacements = (
        ('cls.basic_instr.append(cls.instr_category["SYNCH"])',
         'cls.basic_instr.extend(cls.instr_category["SYNCH"])'),
        ('cls.basic_instr.append(cls.instr_category["CSR"])',
         'cls.basic_instr.extend(cls.instr_category["CSR"])'),
        ('cfg.init_privileged_mode == "MACHINE_MODE"',
         'cfg.init_privileged_mode.name == "MACHINE_MODE"'),
        ('cfg.init_privileged_mode == "SUPERVISOR_MODE"',
         'cfg.init_privileged_mode.name == "SUPERVISOR_MODE"'),
        ('if self.category != riscv_instr_category_t.CSR:',
         'if self.category.get_val() != riscv_instr_category_t.CSR:'),
    )
    expected_counts = (1, 1, 2, 1, 1)
    for (old, new), expected in zip(replacements, expected_counts):
        if source.count(old) != expected:
            raise ValueError("RISC-V-DV CSR instruction patch context changed")
        source = source.replace(old, new)
    csr_asm = (
        ("asm_str = '{} {}, 0x{}, {}'.format(\n"
         "                        asm_str, self.rd.name, self.csr, self.get_imm())",
         "asm_str = '{} {}, {}, {}'.format(\n"
         "                        asm_str, self.rd.name, hex(int(self.csr)), self.get_imm())"),
        ("asm_str = '{} {}, 0x{}, {}'.format(\n"
         "                        asm_str, self.rd.name, self.csr, self.rs1.name)",
         "asm_str = '{} {}, {}, {}'.format(\n"
         "                        asm_str, self.rd.name, hex(int(self.csr)), self.rs1.name)"),
    )
    for old, new in csr_asm:
        if source.count(old) != 1:
            raise ValueError("RISC-V-DV CSR assembly patch context changed")
        source = source.replace(old, new, 1)
    return source


def _patch_riscv_instr_stream(source: str) -> str:
    old_import = (
        "riscv_instr_category_t, riscv_instr_format_t, riscv_reg_t\n"
        "from pygen_src.isa.riscv_instr"
    )
    new_import = (
        "riscv_instr_category_t, riscv_instr_format_t, riscv_reg_t, privileged_reg_t\n"
        "from pygen_src.isa.riscv_instr"
    )
    if source.count(old_import) != 1:
        raise ValueError("RISC-V-DV CSR stream import context changed")
    source = source.replace(old_import, new_import, 1)
    old_return = "        # TODO: Add constraint for CSR, floating point register\n        return instr"
    new_return = """        if (instr.category == riscv_instr_category_t.CSR and
                cfg.no_csr_instr == 0 and not cfg.enable_illegal_csr_instruction):
            csr_names = riscv_instr.include_reg
            if not csr_names:
                raise RuntimeError("RISC-V-DV has no legal CSR selected for this mode")
            csr_name = random.choice(csr_names)
            if csr_name in privileged_reg_t.__members__:
                instr.csr = int(privileged_reg_t[csr_name])
            else:
                raise RuntimeError("RISC-V-DV selected an unknown CSR: {}".format(csr_name))
        return instr"""
    if source.count(old_return) != 1:
        raise ValueError("RISC-V-DV CSR randomization patch context changed")
    return source.replace(old_return, new_return, 1)


def _patch_link_register_config(source: str) -> str:
    old = """    def ra_c(self):
        self.ra != riscv_reg_t.SP
        self.ra != riscv_reg_t.TP
        self.ra != riscv_reg_t.ZERO"""
    new = """    def ra_c(self):
        self.ra != riscv_reg_t.SP
        self.ra != riscv_reg_t.TP
        self.ra != riscv_reg_t.GP
        self.ra != riscv_reg_t.ZERO"""
    if source.count(old) != 1:
        raise ValueError("RISC-V-DV link-register patch context changed")
    return source.replace(old, new, 1)


def _patch_directed_instr_lib(source: str) -> str:
    old = "self.pop_stack_instr = [0] * (self.num_of_reg_to_save + 1)"
    new = (
        "self.pop_stack_instr = [riscv_instr() "
        "for _ in range(self.num_of_reg_to_save + 1)]"
    )
    if source.count(old) != 1:
        raise ValueError("RISC-V-DV stack-pop patch context changed")
    source = source.replace(old, new, 1)
    replacements = (
        ("            self.push_stack_instr[i + 1].process_load_store = 0",
         "            self.push_stack_instr[i + 1].imm_str = str((rcs.XLEN // 8) * (i + 1))\n"
         "            self.push_stack_instr[i + 1].process_load_store = 0\n"
         "            self.push_stack_instr[i + 1].rs1 = cfg.sp\n"
         "            self.push_stack_instr[i + 1].rs2 = self.saved_regs[i]"),
        ("            self.pop_stack_instr[i].process_load_store = 0",
         "            self.pop_stack_instr[i].imm_str = str((rcs.XLEN // 8) * (i + 1))\n"
         "            self.pop_stack_instr[i].process_load_store = 0\n"
         "            self.pop_stack_instr[i].rs1 = cfg.sp\n"
         "            self.pop_stack_instr[i].rd = self.saved_regs[i]"),
    )
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError("RISC-V-DV stack offset patch context changed")
        source = source.replace(old, new, 1)
    source = source.replace(
        "        self.push_stack_instr[0].imm_str = '-{}'.format(self.stack_len)",
        "        self.push_stack_instr[0].imm_str = '-{}'.format(self.stack_len)\n"
        "        self.push_stack_instr[0].rd = cfg.sp\n"
        "        self.push_stack_instr[0].rs1 = cfg.sp",
        1,
    )
    source = source.replace(
        "        self.pop_stack_instr[self.num_of_reg_to_save].imm_str = pkg_ins.format_string(\n"
        "            '{}'.format(self.stack_len))",
        "        self.pop_stack_instr[self.num_of_reg_to_save].imm_str = pkg_ins.format_string(\n"
        "            '{}'.format(self.stack_len))\n"
        "        self.pop_stack_instr[self.num_of_reg_to_save].rd = cfg.sp\n"
        "        self.pop_stack_instr[self.num_of_reg_to_save].rs1 = cfg.sp",
        1,
    )
    push_mix = "        self.mix_instr_stream(self.push_stack_instr)"
    if source.count(push_mix) != 1:
        raise ValueError("RISC-V-DV stack-push placement context changed")
    # Keep the link-register save adjacent to the stack adjustment.  Mixing
    # it with random helper instructions lets a forward branch skip the save,
    # so the generated return can jump through a zero link value.
    source = source.replace(
        push_mix,
        "        self.instr_list = self.push_stack_instr + self.instr_list",
        1,
    )
    return source


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_compat.py RISCV_DV_ROOT [run.py arguments...]")
    root = Path(sys.argv[1]).resolve()
    compat_path = str(Path(__file__).resolve().parent)
    overlay = Path(tempfile.mkdtemp(prefix="rq1-rvdv-pygen-", dir="/tmp"))
    # The generator is executed in-process; clean the per-run namespace after
    # run.py returns so repeated trials do not accumulate stale overlays.
    atexit.register(shutil.rmtree, overlay, ignore_errors=True)
    module_dir = overlay / "pygen_src"
    module_dir.mkdir()
    base_test = root / "pygen" / "pygen_src" / "test" / "riscv_instr_base_test.py"
    base_test_text = _patch_worker_seed(base_test.read_text(encoding="utf-8"))
    source = root / "pygen" / "pygen_src" / "riscv_asm_program_gen.py"
    text = source.read_text(encoding="utf-8")
    sequence = root / "pygen" / "pygen_src" / "riscv_instr_sequence.py"
    sequence_text = sequence.read_text(encoding="utf-8")
    instr = root / "pygen" / "pygen_src" / "isa" / "riscv_instr.py"
    instr_text = _patch_riscv_instr(instr.read_text(encoding="utf-8"))
    instr_stream = root / "pygen" / "pygen_src" / "riscv_instr_stream.py"
    instr_stream_text = _patch_riscv_instr_stream(
        instr_stream.read_text(encoding="utf-8")
    )
    config = root / "pygen" / "pygen_src" / "riscv_instr_gen_config.py"
    config_text = _patch_reserved_regs(
        _patch_link_register_config(config.read_text(encoding="utf-8"))
    )
    directed = root / "pygen" / "pygen_src" / "riscv_directed_instr_lib.py"
    directed_text = _patch_directed_instr_lib(
        directed.read_text(encoding="utf-8")
    )
    for old_text, new_text in (
        ("addi x{} x{} {}", "addi x{}, x{}, {}"),
        ("jalr x{} x{} 0", "jalr x{}, x{}, 0"),
    ):
        if sequence_text.count(old_text) != 1:
            shutil.rmtree(overlay, ignore_errors=True)
            raise SystemExit("RISC-V-DV assembly patch context changed")
        sequence_text = sequence_text.replace(old_text, new_text, 1)
    try:
        text = _patch_generator_init(text)
        sequence_text = _patch_generator_sequence(sequence_text)
    except ValueError as error:
        shutil.rmtree(overlay, ignore_errors=True)
        raise SystemExit(str(error))
    if text.count(OLD) != 1:
        shutil.rmtree(overlay, ignore_errors=True)
        raise SystemExit("RISC-V-DV call-stack patch context changed")
    if sequence_text.count("    def insert_jump_instr(self):") != 1:
        shutil.rmtree(overlay, ignore_errors=True)
        raise SystemExit("RISC-V-DV jump patch context changed")
    sequence_old = (
        "    def insert_jump_instr(self):\n"
        "        # TODO riscv_jump_instr class implementation\n"
        "        \"\"\"\n"
        "        jump_instr = riscv_jump_instr()\n"
        "        jump_instr.target_program_label = target_label\n"
        "        if(not self.is_main_program):\n"
        "            jump_instr.stack_exit_instr = self.instr_stack_exit.pop_stack_instr\n"
        "        jump_instr.label = self.label_name\n"
        "        jump_instr.idx = idx\n"
        "        jump_instr.use_jalr = self.is_main_program\n"
        "        jump_instr.randomize()\n"
        "        self.instr_stream.insert_instr_stream(jump_instr.instr_list)\n"
        "        logging.info(\"{} -> {}...done\".format(jump_instr.jump.instr_name.name, target_label))\n"
        "        \"\"\"\n"
        "        pass\n"
    )
    if sequence_text.count(sequence_old) != 1:
        shutil.rmtree(overlay, ignore_errors=True)
        raise SystemExit("RISC-V-DV jump patch context changed")
    sequence_text = sequence_text.replace(sequence_old, SEQUENCE_JUMP, 1)
    (module_dir / source.name).write_text(text.replace(OLD, NEW), encoding="utf-8")
    (module_dir / sequence.name).write_text(sequence_text, encoding="utf-8")
    isa_dir = module_dir / "isa"
    isa_dir.mkdir()
    (isa_dir / instr.name).write_text(instr_text, encoding="utf-8")
    (module_dir / instr_stream.name).write_text(instr_stream_text, encoding="utf-8")
    (module_dir / config.name).write_text(config_text, encoding="utf-8")
    (module_dir / directed.name).write_text(directed_text, encoding="utf-8")
    test_dir = module_dir / "test"
    test_dir.mkdir()
    (test_dir / base_test.name).write_text(base_test_text, encoding="utf-8")
    (module_dir / "riscv_callstack_gen.py").write_text(CALLSTACK, encoding="utf-8")
    simulator_yaml = overlay / "simulator.yaml"
    simulator_text = (root / "yaml" / "simulator.yaml").read_text(encoding="utf-8")
    try:
        simulator_text = _patch_pyflow_command(simulator_text, test_dir)
    except ValueError as error:
        shutil.rmtree(overlay, ignore_errors=True)
        raise SystemExit(str(error))
    simulator_yaml.write_text(simulator_text, encoding="utf-8")
    overlay_path = str(overlay)
    os.environ["PYTHONPATH"] = os.pathsep.join(
        filter(None, (overlay_path, compat_path, os.environ.get("PYTHONPATH")))
    )
    arguments = list(sys.argv[2:])
    if not any(
        item == "--simulator_yaml" or item.startswith("--simulator_yaml=")
        for item in arguments
    ):
        arguments.extend(("--simulator_yaml", str(simulator_yaml)))
    sys.argv = [str(root / "run.py"), *arguments]
    os.chdir(root)
    # ``pygen_src`` is a namespace package in the pinned snapshot.  Inserting
    # the original ``pygen`` directory at index zero after setting
    # PYTHONPATH makes Python resolve the unpatched modules first, so the
    # compatibility code silently becomes a no-op.  Keep the overlay parent
    # first and retain the original paths as fallback portions of the
    # namespace package.
    search_roots = [
        overlay_path, compat_path, str(root), str(root / "pygen"),
        str(root / ".analysis-out" / "rdv-deps"),
        str(root / "pygen" / "pygen_src"),
    ]
    sys.path[:] = search_roots + [path for path in sys.path if path not in search_roots]
    if os.environ.get("RQ1_RVDV_CONTINUOUS_GENERATION") == "1":
        from scripts import lib as riscv_dv_lib

        upstream_run_cmd = riscv_dv_lib.run_cmd

        def run_generator_without_deadline(cmd, timeout_s=None, *args, **kwargs):
            return upstream_run_cmd(cmd, None, *args, **kwargs)

        riscv_dv_lib.run_cmd = run_generator_without_deadline
    exit_code = 0
    try:
        runpy.run_path(str(root / "run.py"), run_name="__main__")
    except SystemExit as error:
        # The pinned run.py exits after writing its assembly.  Delay that
        # status until the compatibility post-processing has run.
        exit_code = error.code if isinstance(error.code, int) else 0
    # run.py returns only after the assembly files are written.  Repair the
    # final text before the caller performs the bare/Linux cross-build.
    output_arg = next(
        (sys.argv[index + 1] for index, value in enumerate(sys.argv[:-1])
         if value == "--output"),
        None,
    )
    if output_arg:
        _repair_local_branch_references(Path(output_arg))
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
