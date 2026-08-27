#ifndef PICOGRAD_THREAD_POOL_H
#define PICOGRAD_THREAD_POOL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Zero-dependency pthreads pool. Auto-initializes on first use.
// For small n (< threshold) runs serially to avoid overhead.

typedef void (*pg_parallel_fn)(void *ctx, size_t start, size_t end);

// Initialize with nthreads (0 = auto = num cores). Returns 0 on success.
int pg_thread_pool_init(int nthreads);
// Shutdown pool (optional, called at exit)
void pg_thread_pool_fini(void);
int pg_thread_pool_size(void);

// Run fn(ctx, start, end) in parallel over [0, n). Blocks until done.
// If n < threshold or pool size 1, runs serially.
void pg_parallel_for(size_t n, size_t threshold, pg_parallel_fn fn, void *ctx);

// Convenience: get number of hardware threads
int pg_hardware_concurrency(void);

#ifdef __cplusplus
}
#endif

#endif
