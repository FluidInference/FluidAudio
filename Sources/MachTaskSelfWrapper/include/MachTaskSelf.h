// mach_task_self_ is a C macro that Swift 6 can't import directly.
// This wrapper provides a simple function that Swift can safely call.
#ifndef MachTaskSelf_h
#define MachTaskSelf_h

#include <mach/mach.h>

mach_port_t get_current_task_port(void);

#endif /* MachTaskSelf_h */
