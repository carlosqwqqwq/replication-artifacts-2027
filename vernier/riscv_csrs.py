from __future__ import annotations


def _csr_table(data: str) -> dict[int, str]:
    fields = data.split()
    return {int(address, 16): f'"{name}"' for address, name in zip(fields[::2], fields[1::2])}


OFFICIAL_CSRS = _csr_table(
    """
    001 fflags 002 frm 003 fcsr 008 vstart 009 vxsat 00a vxrm 00f vcsr 011 ssp
    015 seed 017 jvt c00 cycle c01 time c02 instret c03 hpmcounter3 c04 hpmcounter4 c05 hpmcounter5
    c06 hpmcounter6 c07 hpmcounter7 c08 hpmcounter8 c09 hpmcounter9 c0a hpmcounter10 c0b hpmcounter11 c0c hpmcounter12 c0d hpmcounter13
    c0e hpmcounter14 c0f hpmcounter15 c10 hpmcounter16 c11 hpmcounter17 c12 hpmcounter18 c13 hpmcounter19 c14 hpmcounter20 c15 hpmcounter21
    c16 hpmcounter22 c17 hpmcounter23 c18 hpmcounter24 c19 hpmcounter25 c1a hpmcounter26 c1b hpmcounter27 c1c hpmcounter28 c1d hpmcounter29
    c1e hpmcounter30 c1f hpmcounter31 c20 vl c21 vtype c22 vlenb 100 sstatus 104 sie 105 stvec
    106 scounteren 10a senvcfg 10c sstateen0 10d sstateen1 10e sstateen2 10f sstateen3 120 scountinhibit 140 sscratch
    141 sepc 142 scause 143 stval 144 sip 14d stimecmp 14e sctrctl 14f sctrstatus 150 siselect
    151 sireg 152 sireg2 153 sireg3 155 sireg4 156 sireg5 157 sireg6 15c stopei 15f sctrdepth
    180 satp 181 srmcfg 183 spmpen 5a8 scontext 200 vsstatus 204 vsie 205 vstvec 240 vsscratch
    241 vsepc 242 vscause 243 vstval 244 vsip 24d vstimecmp 24e vsctrctl 250 vsiselect 251 vsireg
    252 vsireg2 253 vsireg3 255 vsireg4 256 vsireg5 257 vsireg6 25c vstopei 280 vsatp 600 hstatus
    602 hedeleg 603 hideleg 604 hie 605 htimedelta 606 hcounteren 607 hgeie 608 hvien 609 hvictl
    60a henvcfg 60c hstateen0 60d hstateen1 60e hstateen2 60f hstateen3 643 htval 644 hip 645 hvip
    646 hviprio1 647 hviprio2 64a htinst 680 hgatp 6a8 hcontext e12 hgeip eb0 vstopi da0 scountovf
    db0 stopi 007 utvt 045 unxti 046 uintstatus 048 uscratchcsw 049 uscratchcswl 107 stvt 145 snxti
    146 sintstatus 148 sscratchcsw 149 sscratchcswl 307 mtvt 316 mpmpdeleg 345 mnxti 346 mintstatus 348 mscratchcsw
    349 mscratchcswl 300 mstatus 301 misa 302 medeleg 303 mideleg 304 mie 305 mtvec 306 mcounteren
    308 mvien 309 mvip 30a menvcfg 30c mstateen0 30d mstateen1 30e mstateen2 30f mstateen3 320 mcountinhibit
    340 mscratch 341 mepc 342 mcause 343 mtval 344 mip 34a mtinst 34b mtval2 34e mctrctl
    740 mnscratch 741 mnepc 742 mncause 744 mnstatus
    350 miselect 351 mireg 352 mireg2 353 mireg3 355 mireg4 356 mireg5 357 mireg6 35c mtopei
    3a0 pmpcfg0 3a1 pmpcfg1 3a2 pmpcfg2 3a3 pmpcfg3 3a4 pmpcfg4 3a5 pmpcfg5 3a6 pmpcfg6 3a7 pmpcfg7
    3a8 pmpcfg8 3a9 pmpcfg9 3aa pmpcfg10 3ab pmpcfg11 3ac pmpcfg12 3ad pmpcfg13 3ae pmpcfg14 3af pmpcfg15
    3b0 pmpaddr0 3b1 pmpaddr1 3b2 pmpaddr2 3b3 pmpaddr3 3b4 pmpaddr4 3b5 pmpaddr5 3b6 pmpaddr6 3b7 pmpaddr7
    3b8 pmpaddr8 3b9 pmpaddr9 3ba pmpaddr10 3bb pmpaddr11 3bc pmpaddr12 3bd pmpaddr13 3be pmpaddr14 3bf pmpaddr15
    3c0 pmpaddr16 3c1 pmpaddr17 3c2 pmpaddr18 3c3 pmpaddr19 3c4 pmpaddr20 3c5 pmpaddr21 3c6 pmpaddr22 3c7 pmpaddr23
    3c8 pmpaddr24 3c9 pmpaddr25 3ca pmpaddr26 3cb pmpaddr27 3cc pmpaddr28 3cd pmpaddr29 3ce pmpaddr30 3cf pmpaddr31
    3d0 pmpaddr32 3d1 pmpaddr33 3d2 pmpaddr34 3d3 pmpaddr35 3d4 pmpaddr36 3d5 pmpaddr37 3d6 pmpaddr38 3d7 pmpaddr39
    3d8 pmpaddr40 3d9 pmpaddr41 3da pmpaddr42 3db pmpaddr43 3dc pmpaddr44 3dd pmpaddr45 3de pmpaddr46 3df pmpaddr47
    3e0 pmpaddr48 3e1 pmpaddr49 3e2 pmpaddr50 3e3 pmpaddr51 3e4 pmpaddr52 3e5 pmpaddr53 3e6 pmpaddr54 3e7 pmpaddr55
    3e8 pmpaddr56 3e9 pmpaddr57 3ea pmpaddr58 3eb pmpaddr59 3ec pmpaddr60 3ed pmpaddr61 3ee pmpaddr62 3ef pmpaddr63
    747 mseccfg 7a0 tselect 7a1 tdata1 7a2 tdata2 7a3 tdata3 7a4 tinfo 7a5 tcontrol 7a8 mcontext
    7aa mscontext 7b0 dcsr 7b1 dpc 7b2 dscratch0 7b3 dscratch1 b00 mcycle b02 minstret b03 mhpmcounter3
    b04 mhpmcounter4 b05 mhpmcounter5 b06 mhpmcounter6 b07 mhpmcounter7 b08 mhpmcounter8 b09 mhpmcounter9 b0a mhpmcounter10 b0b mhpmcounter11
    b0c mhpmcounter12 b0d mhpmcounter13 b0e mhpmcounter14 b0f mhpmcounter15 b10 mhpmcounter16 b11 mhpmcounter17 b12 mhpmcounter18 b13 mhpmcounter19
    b14 mhpmcounter20 b15 mhpmcounter21 b16 mhpmcounter22 b17 mhpmcounter23 b18 mhpmcounter24 b19 mhpmcounter25 b1a mhpmcounter26 b1b mhpmcounter27
    b1c mhpmcounter28 b1d mhpmcounter29 b1e mhpmcounter30 b1f mhpmcounter31 321 mcyclecfg 322 minstretcfg 323 mhpmevent3 324 mhpmevent4
    325 mhpmevent5 326 mhpmevent6 327 mhpmevent7 328 mhpmevent8 329 mhpmevent9 32a mhpmevent10 32b mhpmevent11 32c mhpmevent12
    32d mhpmevent13 32e mhpmevent14 32f mhpmevent15 330 mhpmevent16 331 mhpmevent17 332 mhpmevent18 333 mhpmevent19 334 mhpmevent20
    335 mhpmevent21 336 mhpmevent22 337 mhpmevent23 338 mhpmevent24 339 mhpmevent25 33a mhpmevent26 33b mhpmevent27 33c mhpmevent28
    33d mhpmevent29 33e mhpmevent30 33f mhpmevent31 f11 mvendorid f12 marchid f13 mimpid f14 mhartid f15 mconfigptr
    fb0 mtopi
    """
)

OFFICIAL_RV32_ONLY_CSRS = _csr_table(
    """
    114 sieh 154 siph 15d stimecmph 193 spmpenh 214 vsieh 254 vsiph 25d vstimecmph 312 medelegh
    612 hedelegh 615 htimedeltah 613 hidelegh 618 hvienh 61a henvcfgh 655 hviph 656 hviprio1h 657 hviprio2h
    61c hstateen0h 61d hstateen1h 61e hstateen2h 61f hstateen3h c80 cycleh c81 timeh c82 instreth c83 hpmcounter3h
    c84 hpmcounter4h c85 hpmcounter5h c86 hpmcounter6h c87 hpmcounter7h c88 hpmcounter8h c89 hpmcounter9h c8a hpmcounter10h c8b hpmcounter11h
    c8c hpmcounter12h c8d hpmcounter13h c8e hpmcounter14h c8f hpmcounter15h c90 hpmcounter16h c91 hpmcounter17h c92 hpmcounter18h c93 hpmcounter19h
    c94 hpmcounter20h c95 hpmcounter21h c96 hpmcounter22h c97 hpmcounter23h c98 hpmcounter24h c99 hpmcounter25h c9a hpmcounter26h c9b hpmcounter27h
    c9c hpmcounter28h c9d hpmcounter29h c9e hpmcounter30h c9f hpmcounter31h 310 mstatush 313 midelegh 314 mieh 318 mvienh
    319 mviph 31a menvcfgh 31c mstateen0h 31d mstateen1h 31e mstateen2h 31f mstateen3h 354 miph 721 mcyclecfgh
    722 minstretcfgh 723 mhpmevent3h 724 mhpmevent4h 725 mhpmevent5h 726 mhpmevent6h 727 mhpmevent7h 728 mhpmevent8h 729 mhpmevent9h
    72a mhpmevent10h 72b mhpmevent11h 72c mhpmevent12h 72d mhpmevent13h 72e mhpmevent14h 72f mhpmevent15h 730 mhpmevent16h 731 mhpmevent17h
    732 mhpmevent18h 733 mhpmevent19h 734 mhpmevent20h 735 mhpmevent21h 736 mhpmevent22h 737 mhpmevent23h 738 mhpmevent24h 739 mhpmevent25h
    73a mhpmevent26h 73b mhpmevent27h 73c mhpmevent28h 73d mhpmevent29h 73e mhpmevent30h 73f mhpmevent31h 757 mseccfgh b80 mcycleh b82 minstreth b83 mhpmcounter3h b84 mhpmcounter4h b85 mhpmcounter5h
    b86 mhpmcounter6h b87 mhpmcounter7h b88 mhpmcounter8h b89 mhpmcounter9h b8a mhpmcounter10h b8b mhpmcounter11h b8c mhpmcounter12h b8d mhpmcounter13h
    b8e mhpmcounter14h b8f mhpmcounter15h b90 mhpmcounter16h b91 mhpmcounter17h b92 mhpmcounter18h b93 mhpmcounter19h b94 mhpmcounter20h b95 mhpmcounter21h
    b96 mhpmcounter22h b97 mhpmcounter23h b98 mhpmcounter24h b99 mhpmcounter25h b9a mhpmcounter26h b9b mhpmcounter27h b9c mhpmcounter28h b9d mhpmcounter29h
    b9e mhpmcounter30h b9f mhpmcounter31h
    """
)

def csr_name(address: int) -> str | None:
    name = OFFICIAL_CSRS.get(int(address)) or OFFICIAL_RV32_ONLY_CSRS.get(int(address))
    return None if name is None else name.strip('"')


def csr_is_read_only(address: int) -> bool:
    address = int(address)
    return 0 <= address <= 0xFFF and (address >> 10) & 0x3 == 0x3


def csr_minimum_privilege(address: int) -> int:
    address = int(address)
    return 0 if not 0 <= address <= 0xFFF else (address >> 8) & 0x3


def csr_is_rv32_only(address: int) -> bool:
    return int(address) in OFFICIAL_RV32_ONLY_CSRS


def csr_required_extensions(address: int) -> frozenset[str]:
    name = csr_name(address) or ""
    if name in {"mseccfg", "mseccfgh"}:
        return frozenset({"smepmp"})
    if name == "srmcfg":
        return frozenset({"ssqosid"})
    if name in {
        "scontext", "hcontext", "mcontext", "mscontext",
        "tselect", "tdata1", "tdata2", "tdata3", "tinfo", "tcontrol",
    }:
        return frozenset({"h", "sdtrig"}) if name == "hcontext" else frozenset({"sdtrig"})
    if name.startswith(("siselect", "sireg")):
        return frozenset({"sscsrind"})
    if name.startswith(("vsiselect", "vsireg")):
        return frozenset({"h", "sscsrind"})
    if name.startswith(("miselect", "mireg")):
        return frozenset({"smcsrind"})
    if name in {"mnscratch", "mnepc", "mncause", "mnstatus"}:
        return frozenset({"smrnmi"})
    if name.startswith("mhpmevent") and name.endswith("h"):
        return frozenset({"sscofpmf"})
    if name in {"stimecmp", "stimecmph", "vstimecmp", "vstimecmph"}:
        return frozenset({"sstc"})
    if name in {
        "mstateen0", "mstateen1", "mstateen2", "mstateen3",
        "mstateen0h", "mstateen1h", "mstateen2h", "mstateen3h",
        "sstateen0", "sstateen1", "sstateen2", "sstateen3",
        "hstateen0", "hstateen1", "hstateen2", "hstateen3",
        "hstateen0h", "hstateen1h", "hstateen2h", "hstateen3h",
    }:
        return frozenset({"smstateen"})
    if name == "mctrctl":
        return frozenset({"smctr"})
    if name in {"mcyclecfg", "minstretcfg", "mcyclecfgh", "minstretcfgh"}:
        return frozenset({"smcntrpmf"})
    if name == "scountovf":
        return frozenset({"sscofpmf"})
    if name == "scountinhibit":
        return frozenset({"smcdeleg", "ssccfg"})
    if name in {
        "hstatus", "hedeleg", "hideleg", "hie", "htimedelta", "htimedeltah",
        "hcounteren", "hgeie", "hgeip", "htval", "hip", "hvip", "htinst",
        "hgatp", "hedelegh",
        "vsstatus", "vsie", "vstvec", "vsscratch", "vsepc", "vscause",
        "vstval", "vsip", "vsatp",
    }:
        return frozenset({"h"})
    if name in {"cycle", "time", "instret", "cycleh", "timeh", "instreth"}:
        return frozenset({"zicntr"})
    if name.startswith("hpmcounter"):
        return frozenset({"zihpm"})
    if name in {"sctrctl", "sctrstatus", "sctrdepth", "vsctrctl"}:
        return frozenset({"ssctr"})
    if name in {"fflags", "frm", "fcsr"}:
        return frozenset({"f"})
    if name in {"vl", "vtype", "vlenb", "vstart", "vxsat", "vxrm", "vcsr"}:
        return frozenset({"v"})
    if name == "jvt":
        return frozenset({"zcmt"})
    if name == "seed":
        return frozenset({"zkr"})
    if name == "ssp":
        return frozenset({"zicfiss"})
    return frozenset()
