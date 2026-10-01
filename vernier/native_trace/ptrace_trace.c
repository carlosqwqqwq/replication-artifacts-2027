#include <errno.h>
#include <elf.h>
#include <inttypes.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ptrace.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <sys/wait.h>
#include <unistd.h>

#include <asm/ptrace.h>

#define RV_EBREAK 0x00100073UL
#define RV_C_EBREAK 0x9002U

struct pc_entry {
    unsigned long pc;
    unsigned int length;
    unsigned char active;
    unsigned char exit_sentinel;
};

struct pc_set {
    struct pc_entry *items;
    size_t count;
};

struct bp_slot {
    unsigned long aligned_address;
    unsigned long original_word;
};

struct bp_table {
    struct bp_slot *slots;
    size_t count;
};

struct fp_snapshot {
    union __riscv_fp_state state;
    size_t bytes;
    int available;
};

struct pending_checkpoint {
    struct user_regs_struct before;
    struct fp_snapshot before_fp;
    unsigned long pc;
    int valid;
};

static int read_regs(pid_t child, struct user_regs_struct *regs) {
    struct iovec iov;
    memset(regs, 0, sizeof(*regs));
    iov.iov_base = regs;
    iov.iov_len = sizeof(*regs);
    if (ptrace(PTRACE_GETREGSET, child, (void *)(uintptr_t)NT_PRSTATUS, &iov) != 0) {
        fprintf(stderr, "RV_TRACE_ERROR=ptrace-getregset:%s\n", strerror(errno));
        return -1;
    }
    return 0;
}

static int read_fpregs(pid_t child, struct fp_snapshot *snapshot) {
    struct iovec iov;
    memset(snapshot, 0, sizeof(*snapshot));
    iov.iov_base = &snapshot->state;
    iov.iov_len = sizeof(snapshot->state);
    if (ptrace(PTRACE_GETREGSET, child, (void *)(uintptr_t)NT_PRFPREG, &iov) != 0) {
        return -1;
    }
    snapshot->bytes = iov.iov_len;
    snapshot->available = 1;
    return 0;
}

static int wait_child(pid_t child, int *status) {
    for (;;) {
        if (waitpid(child, status, 0) >= 0) {
            return 0;
        }
        if (errno != EINTR) {
            fprintf(stderr, "RV_TRACE_ERROR=waitpid:%s\n", strerror(errno));
            return -1;
        }
    }
}

static int read_text_word(pid_t child, unsigned long address, unsigned long *word) {
    errno = 0;
    long value = ptrace(PTRACE_PEEKTEXT, child, (void *)address, 0);
    if (value == -1 && errno != 0) {
        fprintf(stderr, "RV_TRACE_ERROR=ptrace-peektext:%s\n", strerror(errno));
        return -1;
    }
    *word = (unsigned long)value;
    return 0;
}

static int write_text_word(pid_t child, unsigned long address, unsigned long word) {
    if (ptrace(PTRACE_POKETEXT, child, (void *)address, (void *)word) != 0) {
        fprintf(stderr, "RV_TRACE_ERROR=ptrace-poketext:%s\n", strerror(errno));
        return -1;
    }
    return 0;
}

static int finish_child(pid_t child, int status) {
    if (!WIFEXITED(status) && !WIFSIGNALED(status)) {
        if (ptrace(PTRACE_CONT, child, 0, 0) != 0) {
            fprintf(stderr, "RV_TRACE_ERROR=ptrace-cont:%s\n", strerror(errno));
            return 127;
        }
        for (;;) {
            if (wait_child(child, &status) != 0) {
                return 127;
            }
            if (WIFEXITED(status) || WIFSIGNALED(status)) {
                break;
            }
            if (WIFSTOPPED(status)) {
                int sig = WSTOPSIG(status);
                if (ptrace(PTRACE_CONT, child, 0, (void *)(uintptr_t)sig) != 0) {
                    fprintf(stderr, "RV_TRACE_ERROR=ptrace-cont-signal:%s\n", strerror(errno));
                    return 127;
                }
            }
        }
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
}

static unsigned long parse_pc(const char *text, const char *name) {
    char *end = NULL;
    errno = 0;
    if (text[0] == '-') {
        fprintf(stderr, "RV_TRACE_ERROR=invalid-%s\n", name);
        exit(2);
    }
    unsigned long value = strtoul(text, &end, 0);
    if (errno != 0 || end == text) {
        fprintf(stderr, "RV_TRACE_ERROR=invalid-%s\n", name);
        exit(2);
    }
    while (*end == ' ' || *end == '\t' || *end == '\r' || *end == '\n') {
        end++;
    }
    if (*end != '\0') {
        fprintf(stderr, "RV_TRACE_ERROR=invalid-%s\n", name);
        exit(2);
    }
    if ((value & 0x1UL) != 0) {
        fprintf(stderr, "RV_TRACE_ERROR=unaligned-pc\n");
        exit(2);
    }
    return value;
}

static unsigned int parse_length(const char *text) {
    char *end = NULL;
    errno = 0;
    unsigned long value = strtoul(text, &end, 0);
    if (errno != 0 || end == text) {
        fprintf(stderr, "RV_TRACE_ERROR=invalid-pc-length\n");
        exit(2);
    }
    while (*end == ' ' || *end == '\t' || *end == '\r' || *end == '\n') {
        end++;
    }
    if (*end != '\0' || (value != 2UL && value != 4UL)) {
        fprintf(stderr, "RV_TRACE_ERROR=invalid-pc-length\n");
        exit(2);
    }
    return (unsigned int)value;
}

static int pc_compare(const void *left, const void *right) {
    const struct pc_entry *a = (const struct pc_entry *)left;
    const struct pc_entry *b = (const struct pc_entry *)right;
    if (a->pc == b->pc) {
        return (a->length > b->length) - (a->length < b->length);
    }
    unsigned long left_pc = a->pc;
    unsigned long right_pc = b->pc;
    return (left_pc > right_pc) - (left_pc < right_pc);
}

static size_t slot_count_for_entry(const struct pc_entry *entry) {
    unsigned long aligned = entry->pc & ~(sizeof(unsigned long) - 1UL);
    unsigned long offset = entry->pc - aligned;
    return offset + entry->length > sizeof(unsigned long) ? 2U : 1U;
}

static int slot_touches_entry(const struct bp_slot *slot, const struct pc_entry *entry) {
    unsigned long slot_start = slot->aligned_address;
    unsigned long slot_stop = slot_start + sizeof(unsigned long);
    unsigned long entry_start = entry->pc;
    unsigned long entry_stop = entry_start + entry->length;
    return entry_start < slot_stop && entry_stop > slot_start;
}

static void word_to_bytes(unsigned long word, unsigned char out[sizeof(unsigned long)]) {
    for (size_t index = 0; index < sizeof(unsigned long); index++) {
        out[index] = (unsigned char)((word >> (index * 8U)) & 0xffU);
    }
}

static unsigned long bytes_to_word(const unsigned char bytes[sizeof(unsigned long)]) {
    unsigned long value = 0;
    for (size_t index = 0; index < sizeof(unsigned long); index++) {
        value |= ((unsigned long)bytes[index]) << (index * 8U);
    }
    return value;
}

static unsigned char original_byte(const struct bp_table *table, unsigned long address) {
    unsigned long aligned = address & ~(sizeof(unsigned long) - 1UL);
    for (size_t index = 0; index < table->count; index++) {
        if (table->slots[index].aligned_address == aligned) {
            return (unsigned char)(table->slots[index].original_word >> ((address - aligned) * 8U));
        }
    }
    return 0;
}

static int original_instruction_is_ebreak(
    const struct bp_table *table,
    const struct pc_entry *entry
) {
    const unsigned char ebreak[4] = {0x73, 0x00, 0x10, 0x00};
    const unsigned char c_ebreak[2] = {0x02, 0x90};
    const unsigned char *expected = entry->length == 2U ? c_ebreak : ebreak;
    for (unsigned int index = 0; index < entry->length; index++) {
        if (original_byte(table, entry->pc + index) != expected[index]) {
            return 0;
        }
    }
    return 1;
}

static int original_instruction_length_matches(
    const struct bp_table *table,
    const struct pc_entry *entry
) {
    unsigned int first_halfword = original_byte(table, entry->pc)
        | ((unsigned int)original_byte(table, entry->pc + 1U) << 8U);
    if ((first_halfword & 0x3U) != 0x3U) {
        return entry->length == 2U;
    }
    return (first_halfword & 0x1cU) != 0x1cU && entry->length == 4U;
}

static void patch_bytes_for_entry(const struct pc_entry *entry, unsigned char out[4]) {
    if (entry->length == 2U) {
        out[0] = (unsigned char)(RV_C_EBREAK & 0xffU);
        out[1] = (unsigned char)((RV_C_EBREAK >> 8U) & 0xffU);
        return;
    }
    out[0] = (unsigned char)(RV_EBREAK & 0xffU);
    out[1] = (unsigned char)((RV_EBREAK >> 8U) & 0xffU);
    out[2] = (unsigned char)((RV_EBREAK >> 16U) & 0xffU);
    out[3] = (unsigned char)((RV_EBREAK >> 24U) & 0xffU);
}

static struct pc_entry *find_entry(struct pc_set *pcs, unsigned long pc) {
    for (size_t index = 0; index < pcs->count; index++) {
        if (pcs->items[index].pc == pc) {
            return &pcs->items[index];
        }
    }
    return NULL;
}

static void add_slot(struct bp_slot *slots, size_t *count, unsigned long aligned) {
    for (size_t index = 0; index < *count; index++) {
        if (slots[index].aligned_address == aligned) {
            return;
        }
    }
    slots[*count].aligned_address = aligned;
    (*count)++;
}

static struct pc_set load_pc_manifest(const char *path) {
    FILE *file = fopen(path, "r");
    if (file == NULL) {
        fprintf(stderr, "RV_TRACE_ERROR=open-pc-manifest:%s\n", strerror(errno));
        exit(2);
    }
    struct pc_entry *items = NULL;
    size_t count = 0;
    size_t capacity = 0;
    char line[128];
    size_t sentinel_count = 0;
    while (fgets(line, sizeof(line), file) != NULL) {
        char *start = line;
        while (*start == ' ' || *start == '\t') {
            start++;
        }
        if (*start == '\0' || *start == '\r' || *start == '\n' || *start == '#') {
            continue;
        }
        if (count == capacity) {
            if (capacity > SIZE_MAX / 2U) {
                fprintf(stderr, "RV_TRACE_ERROR=pc-manifest-size-overflow\n");
                exit(2);
            }
            size_t next_capacity = capacity ? capacity * 2U : 256U;
            if (next_capacity > SIZE_MAX / sizeof(*items)) {
                fprintf(stderr, "RV_TRACE_ERROR=pc-manifest-size-overflow\n");
                exit(2);
            }
            struct pc_entry *grown = realloc(
                items, next_capacity * sizeof(*items)
            );
            if (grown == NULL) {
                fprintf(stderr, "RV_TRACE_ERROR=alloc-pc-manifest\n");
                exit(2);
            }
            items = grown;
            capacity = next_capacity;
        }
        char pc_text[64] = {0};
        char length_text[32] = {0};
        char marker[32] = {0};
        char extra[2] = {0};
        int fields = sscanf(start, "%63s %31s %31s %1s", pc_text, length_text, marker, extra);
        if (fields < 1 || fields > 3) {
            fprintf(stderr, "RV_TRACE_ERROR=invalid-pc-manifest-entry\n");
            exit(2);
        }
        items[count].pc = parse_pc(pc_text, "pc-manifest-entry");
        items[count].length = 4U;
        items[count].active = 1U;
        items[count].exit_sentinel = 0U;
        if (fields >= 2) {
            items[count].length = parse_length(length_text);
        }
        if (fields == 3) {
            if (strcmp(marker, "exit-sentinel") != 0) {
                fprintf(stderr, "RV_TRACE_ERROR=invalid-pc-manifest-marker\n");
                exit(2);
            }
            items[count].exit_sentinel = 1U;
            sentinel_count++;
        }
        count++;
    }
    fclose(file);
    if (count == 0) {
        fprintf(stderr, "RV_TRACE_ERROR=empty-pc-manifest\n");
        exit(2);
    }
    if (sentinel_count != 1U) {
        fprintf(stderr, "RV_TRACE_ERROR=missing-or-duplicate-exit-sentinel\n");
        exit(2);
    }
    qsort(items, count, sizeof(struct pc_entry), pc_compare);
    size_t unique = 0;
    for (size_t index = 0; index < count; index++) {
        if (unique == 0 || items[index].pc != items[unique - 1].pc) {
            items[unique++] = items[index];
            continue;
        }
        if (items[index].length != items[unique - 1].length) {
            fprintf(stderr, "RV_TRACE_ERROR=duplicate-pc-with-different-length\n");
            exit(2);
        }
        if (items[index].exit_sentinel != items[unique - 1].exit_sentinel) {
            fprintf(stderr, "RV_TRACE_ERROR=duplicate-pc-with-conflicting-marker\n");
            exit(2);
        }
    }
    for (size_t index = 1; index < unique; index++) {
        if (items[index].pc - items[index - 1].pc < items[index - 1].length) {
            fprintf(stderr, "RV_TRACE_ERROR=overlapping-pc-ranges\n");
            exit(2);
        }
    }
    struct pc_set result = {items, unique};
    return result;
}

static struct bp_table make_bp_table(const struct pc_set *pcs) {
    if (pcs->count > SIZE_MAX / 2U) {
        fprintf(stderr, "RV_TRACE_ERROR=pc-manifest-size-overflow\n");
        exit(2);
    }
    struct bp_slot *slots = calloc(pcs->count * 2U, sizeof(struct bp_slot));
    if (slots == NULL) {
        fprintf(stderr, "RV_TRACE_ERROR=alloc-breakpoints\n");
        exit(2);
    }
    size_t count = 0;
    for (size_t index = 0; index < pcs->count; index++) {
        unsigned long aligned = pcs->items[index].pc & ~(sizeof(unsigned long) - 1UL);
        add_slot(slots, &count, aligned);
        if (slot_count_for_entry(&pcs->items[index]) == 2U) {
            add_slot(slots, &count, aligned + sizeof(unsigned long));
        }
    }
    struct bp_table table = {slots, count};
    return table;
}

static unsigned long patched_word_for_slot(
    const struct bp_slot *slot,
    const struct pc_set *pcs
) {
    unsigned char bytes[sizeof(unsigned long)];
    word_to_bytes(slot->original_word, bytes);
    for (size_t index = 0; index < pcs->count; index++) {
        const struct pc_entry *entry = &pcs->items[index];
        if (!entry->active || !slot_touches_entry(slot, entry)) {
            continue;
        }
        unsigned char patch[4] = {0};
        patch_bytes_for_entry(entry, patch);
        unsigned long slot_start = slot->aligned_address;
        unsigned long slot_stop = slot_start + sizeof(unsigned long);
        unsigned long entry_start = entry->pc;
        unsigned long entry_stop = entry_start + entry->length;
        unsigned long overlap_start = entry_start > slot_start ? entry_start : slot_start;
        unsigned long overlap_stop = entry_stop < slot_stop ? entry_stop : slot_stop;
        size_t entry_offset = (size_t)(overlap_start - entry_start);
        size_t slot_offset = (size_t)(overlap_start - slot_start);
        size_t copy_size = (size_t)(overlap_stop - overlap_start);
        memcpy(bytes + slot_offset, patch + entry_offset, copy_size);
    }
    return bytes_to_word(bytes);
}

static int install_breakpoints(pid_t child, struct bp_table *table, const struct pc_set *pcs) {
    for (size_t index = 0; index < table->count; index++) {
        struct bp_slot *slot = &table->slots[index];
        if (read_text_word(child, slot->aligned_address, &slot->original_word) != 0) {
            return -1;
        }
    }
    for (size_t index = 0; index < pcs->count; index++) {
        if (!original_instruction_length_matches(table, &pcs->items[index])) {
            fprintf(stderr, "RV_TRACE_ERROR=pc-manifest-length-mismatch\n");
            return -1;
        }
    }
    for (size_t index = 0; index < table->count; index++) {
        struct bp_slot *slot = &table->slots[index];
        if (write_text_word(child, slot->aligned_address, patched_word_for_slot(slot, pcs)) != 0) {
            return -1;
        }
    }
    return 0;
}

static int set_breakpoint_state(
    pid_t child,
    struct bp_table *table,
    struct pc_set *pcs,
    unsigned long pc,
    unsigned char active
) {
    struct pc_entry *entry = find_entry(pcs, pc);
    if (entry == NULL) {
        fprintf(stderr, "RV_TRACE_ERROR=breakpoint-entry-missing\n");
        return -1;
    }
    if (entry->active == active) {
        fprintf(
            stderr,
            "RV_TRACE_ERROR=%s\n",
            active ? "breakpoint-already-armed" : "breakpoint-already-cleared"
        );
        return -1;
    }
    entry->active = active;
    for (size_t slot_index = 0; slot_index < table->count; slot_index++) {
        struct bp_slot *slot = &table->slots[slot_index];
        if (!slot_touches_entry(slot, entry)) {
            continue;
        }
        if (write_text_word(child, slot->aligned_address, patched_word_for_slot(slot, pcs)) != 0) {
            return -1;
        }
    }
    return 0;
}

static unsigned long gpr_value(const struct user_regs_struct *regs, int index) {
    if (index == 0) {
        return 0UL;
    }
    const unsigned long *words = (const unsigned long *)regs;
    return words[index];
}

static void print_gpr_array(const struct user_regs_struct *regs) {
    fputc('[', stderr);
    for (int index = 0; index < 32; index++) {
        if (index != 0) {
            fputc(',', stderr);
        }
        fprintf(stderr, "\"0x%lx\"", gpr_value(regs, index));
    }
    fputc(']', stderr);
}

static uint64_t fp_rawbits_at(const struct fp_snapshot *snapshot, int index) {
    if (snapshot->bytes >= sizeof(snapshot->state.d)) {
        return (uint64_t)snapshot->state.d.f[index];
    }
    return (uint64_t)snapshot->state.f.f[index];
}

static uint32_t fp_fcsr(const struct fp_snapshot *snapshot) {
    if (snapshot->bytes >= sizeof(snapshot->state.d)) {
        return (uint32_t)snapshot->state.d.fcsr;
    }
    return (uint32_t)snapshot->state.f.fcsr;
}

static void print_fpr_array(const struct fp_snapshot *snapshot) {
    fputc('[', stderr);
    for (int index = 0; index < 32; index++) {
        if (index != 0) {
            fputc(',', stderr);
        }
        fprintf(stderr, "\"0x%" PRIx64 "\"", fp_rawbits_at(snapshot, index));
    }
    fputc(']', stderr);
}

static void print_checkpoint(
    const struct user_regs_struct *before,
    const struct user_regs_struct *after,
    const struct fp_snapshot *before_fp,
    const struct fp_snapshot *after_fp
) {
    fprintf(stderr, "RV_TRACE_CHECKPOINT={\"pc\":\"0x%lx\",\"after_pc\":\"0x%lx\",\"before_gpr\":", before->pc, after->pc);
    print_gpr_array(before);
    fputs(",\"after_gpr\":", stderr);
    print_gpr_array(after);
    if (before_fp != NULL && after_fp != NULL && before_fp->available && after_fp->available) {
        uint32_t before_fcsr = fp_fcsr(before_fp);
        uint32_t after_fcsr = fp_fcsr(after_fp);
        fputs(",\"before_fpr_rawbits\":", stderr);
        print_fpr_array(before_fp);
        fputs(",\"after_fpr_rawbits\":", stderr);
        print_fpr_array(after_fp);
        fprintf(
            stderr,
            ",\"before_fflags\":\"0x%x\",\"after_fflags\":\"0x%x\",\"before_frm\":\"0x%x\",\"after_frm\":\"0x%x\"",
            before_fcsr & 0x1F,
            after_fcsr & 0x1F,
            (before_fcsr >> 5) & 0x7,
            (after_fcsr >> 5) & 0x7
        );
    }
    fputs("}\n", stderr);
}

static int terminal_signal(int sig) {
    return sig == SIGILL || sig == SIGSEGV || sig == SIGBUS ||
           sig == SIGFPE || sig == SIGABRT || sig == SIGSYS;
}

static int kill_child(pid_t child) {
    ptrace(PTRACE_KILL, child, 0, 0);
    return 127;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <elf> <pc-manifest>\n", argv[0]);
        return 2;
    }

    const char *elf_path = argv[1];
    struct pc_set pcs = load_pc_manifest(argv[2]);
    struct bp_table table = make_bp_table(&pcs);

    pid_t child = fork();
    if (child < 0) {
        fprintf(stderr, "RV_TRACE_ERROR=fork:%s\n", strerror(errno));
        return 127;
    }
    if (child == 0) {
        if (ptrace(PTRACE_TRACEME, 0, 0, 0) != 0) {
            fprintf(stderr, "RV_TRACE_ERROR=child-traceme:%s\n", strerror(errno));
            _exit(127);
        }
        execl(elf_path, elf_path, (char *)NULL);
        fprintf(stderr, "RV_TRACE_ERROR=exec:%s\n", strerror(errno));
        _exit(127);
    }

    int status = 0;
    if (wait_child(child, &status) != 0) {
        return 127;
    }
    if (!WIFSTOPPED(status)) {
        return finish_child(child, status);
    }

    if (install_breakpoints(child, &table, &pcs) != 0) {
        return kill_child(child);
    }

    unsigned long hits = 0;
    int unexpected_pc = 0;
    int exit_sentinel_seen = 0;
    int terminal_signal_seen = 0;
    int terminal_signal_code = 0;
    int final_observer_trap = 0;
    unsigned long checkpoints = 0;
    struct pending_checkpoint pending;
    memset(&pending, 0, sizeof(pending));

    if (ptrace(PTRACE_CONT, child, 0, 0) != 0) {
        fprintf(stderr, "RV_TRACE_ERROR=ptrace-cont:%s\n", strerror(errno));
        return kill_child(child);
    }

    for (;;) {
        if (wait_child(child, &status) != 0) {
            return 127;
        }
        if (WIFEXITED(status) || WIFSIGNALED(status)) {
            break;
        }
        if (!WIFSTOPPED(status)) {
            fprintf(stderr, "RV_TRACE_ERROR=unexpected-wait-status\n");
            return kill_child(child);
        }
        if (WSTOPSIG(status) != SIGTRAP) {
            int sig = WSTOPSIG(status);
            if (terminal_signal(sig)) {
                struct user_regs_struct regs;
                if (read_regs(child, &regs) != 0) {
                    return kill_child(child);
                }
                if (pending.valid) {
                    struct fp_snapshot fpregs;
                    if (read_fpregs(child, &fpregs) != 0) {
                        memset(&fpregs, 0, sizeof(fpregs));
                    }
                    print_checkpoint(&pending.before, &regs, &pending.before_fp, &fpregs);
                    pending.valid = 0;
                    checkpoints++;
                }
                terminal_signal_seen = 1;
                terminal_signal_code = sig;
                final_observer_trap = 1;
                fprintf(stderr, "RV_TRACE_TERMINAL_SIGNAL=%d\n", sig);
                fprintf(stderr, "RV_TRACE_FINAL_TRAP_PC=0x%lx\n", regs.pc);
                ptrace(PTRACE_KILL, child, 0, 0);
                if (wait_child(child, &status) != 0) {
                    return 127;
                }
                break;
            }
            if (ptrace(PTRACE_CONT, child, 0, (void *)(uintptr_t)sig) != 0) {
                fprintf(stderr, "RV_TRACE_ERROR=ptrace-cont-signal:%s\n", strerror(errno));
                return kill_child(child);
            }
            continue;
        }

        struct user_regs_struct regs;
        if (read_regs(child, &regs) != 0) {
            return kill_child(child);
        }
        struct fp_snapshot fpregs;
        if (read_fpregs(child, &fpregs) != 0) {
            memset(&fpregs, 0, sizeof(fpregs));
        }
        unsigned long pc = regs.pc;
        struct pc_entry *entry = find_entry(&pcs, pc);
        if (entry == NULL) {
            if (exit_sentinel_seen) {
                fprintf(stderr, "RV_TRACE_FINAL_TRAP_PC=0x%lx\n", pc);
                final_observer_trap = 1;
                ptrace(PTRACE_KILL, child, 0, 0);
                if (wait_child(child, &status) != 0) {
                    return 127;
                }
                break;
            }
            unexpected_pc = 1;
            fprintf(stderr, "RV_TRACE_ERROR=unexpected-breakpoint-pc\n");
            fprintf(stderr, "RV_TRACE_ACTUAL_PC=0x%lx\n", pc);
            return kill_child(child);
        }
        if (original_instruction_is_ebreak(&table, entry)) {
            fprintf(stderr, "RV_TRACE_ERROR=original-ebreak-not-traceable\n");
            return kill_child(child);
        }
        if (!entry->active) {
            fprintf(stderr, "RV_TRACE_ERROR=inactive-breakpoint-hit\n");
            return kill_child(child);
        }
        if (pending.valid) {
            if (set_breakpoint_state(child, &table, &pcs, pending.pc, 1U) != 0) {
                return kill_child(child);
            }
            print_checkpoint(&pending.before, &regs, &pending.before_fp, &fpregs);
            pending.valid = 0;
            checkpoints++;
        }
        if (entry->exit_sentinel) {
            if (exit_sentinel_seen) {
                fprintf(stderr, "RV_TRACE_ERROR=duplicate-exit-sentinel-hit\n");
                return kill_child(child);
            }
            exit_sentinel_seen = 1;
            if (set_breakpoint_state(child, &table, &pcs, pc, 0U) != 0) {
                return kill_child(child);
            }
            if (ptrace(PTRACE_CONT, child, 0, 0) != 0) {
                fprintf(stderr, "RV_TRACE_ERROR=ptrace-cont-after-exit-sentinel:%s\n", strerror(errno));
                return kill_child(child);
            }
            continue;
        }
        fprintf(stderr, "RV_TRACE_PC=0x%lx\n", pc);
        hits++;
        if (set_breakpoint_state(child, &table, &pcs, pc, 0U) != 0) {
            return kill_child(child);
        }
        pending.before = regs;
        pending.before_fp = fpregs;
        pending.pc = pc;
        pending.valid = 1;
        if (ptrace(PTRACE_CONT, child, 0, 0) != 0) {
            fprintf(stderr, "RV_TRACE_ERROR=ptrace-cont-after-hit:%s\n", strerror(errno));
            return kill_child(child);
        }
    }

    int trace_complete = hits > 0 &&
        (exit_sentinel_seen || terminal_signal_seen) &&
        !pending.valid && checkpoints == hits;
    if (!trace_complete) {
        fprintf(stderr, "RV_TRACE_ERROR=incomplete-exit-sentinel-chain\n");
    }
    fprintf(stderr, "RV_TRACE_STATUS=%s\n", trace_complete ? "ok" : "incomplete");
    fprintf(stderr, "RV_TRACE_HITS=%lu\n", hits);
    fprintf(stderr, "RV_TRACE_CHECKPOINTS=%lu\n", checkpoints);
    fprintf(stderr, "RV_TRACE_MANIFEST_COUNT=%zu\n", pcs.count);
    fprintf(stderr, "RV_TRACE_UNEXPECTED_PC=%d\n", unexpected_pc);
    if (terminal_signal_seen) {
        fprintf(stderr, "RV_TRACE_TERMINAL_SIGNAL_CODE=%d\n", terminal_signal_code);
    }

    free(pcs.items);
    free(table.slots);
    return trace_complete ? (final_observer_trap ? 0 : finish_child(child, status)) : 3;
}
