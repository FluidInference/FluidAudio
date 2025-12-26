#ifndef MachTaskSelf_h
#define MachTaskSelf_h

#include <mach/mach.h>

/// Returns the current task port in a way that's safe to call from Swift 6.
mach_port_t get_current_task_port(void);

#endif /* MachTaskSelf_h */
