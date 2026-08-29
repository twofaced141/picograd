#include "pool.h"
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

typedef struct pg_task {
    pg_parallel_fn fn;
    void *ctx;
    size_t start;
    size_t end;
    struct pg_task *next;
} pg_task_t;

typedef struct {
    pthread_mutex_t mu;
    pthread_cond_t cv;
    pthread_cond_t done_cv;
    pg_task_t *head;
    pg_task_t *tail;
    pthread_t *threads;
    int nthreads;
    int active; // tasks in flight
    int shutdown;
} pg_pool_t;

static pg_pool_t g_pool = {
    .mu = PTHREAD_MUTEX_INITIALIZER,
    .cv = PTHREAD_COND_INITIALIZER,
    .done_cv = PTHREAD_COND_INITIALIZER,
    .head = NULL,
    .tail = NULL,
    .threads = NULL,
    .nthreads = 0,
    .active = 0,
    .shutdown = 0,
};
static pthread_once_t g_once = PTHREAD_ONCE_INIT;
static int g_init_done = 0;

int pg_hardware_concurrency(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n < 1) n = 1;
    if (n > 64) n = 64;
    return (int)n;
}

static void *worker_fn(void *arg) {
    (void)arg;
    pg_pool_t *p = &g_pool;
    for (;;) {
        pthread_mutex_lock(&p->mu);
        while (!p->head && !p->shutdown) {
            pthread_cond_wait(&p->cv, &p->mu);
        }
        if (p->shutdown) {
            pthread_mutex_unlock(&p->mu);
            break;
        }
        pg_task_t *task = p->head;
        if (task) {
            p->head = task->next;
            if (!p->head) p->tail = NULL;
            p->active++;
        }
        pthread_mutex_unlock(&p->mu);
        if (task) {
            task->fn(task->ctx, task->start, task->end);
            pthread_mutex_lock(&p->mu);
            p->active--;
            if (!p->head && p->active == 0) {
                pthread_cond_broadcast(&p->done_cv);
            }
            pthread_mutex_unlock(&p->mu);
            free(task);
        }
    }
    return NULL;
}

static void pool_init_once(void) {
    int want = pg_hardware_concurrency();
    // env override
    const char *env = getenv("PICOGRAD_THREADS");
    if (env) {
        int v = atoi(env);
        if (v >= 0) want = v;
    }
    if (want < 1) want = 1;
    // For single core, no threads needed
    if (want == 1) {
        g_pool.nthreads = 1;
        g_init_done = 1;
        return;
    }
    g_pool.nthreads = want;
    g_pool.threads = calloc((size_t)want, sizeof(pthread_t));
    if (!g_pool.threads) {
        g_pool.nthreads = 1;
        g_init_done = 1;
        return;
    }
    pthread_mutex_init(&g_pool.mu, NULL);
    pthread_cond_init(&g_pool.cv, NULL);
    pthread_cond_init(&g_pool.done_cv, NULL);
    for (int i = 0; i < want; i++) {
        int rc = pthread_create(&g_pool.threads[i], NULL, worker_fn, NULL);
        if (rc != 0) {
            // fallback to fewer threads
            g_pool.nthreads = i;
            break;
        }
    }
    if (g_pool.nthreads == 0) {
        g_pool.nthreads = 1;
        free(g_pool.threads);
        g_pool.threads = NULL;
    }
    g_init_done = 1;
}

int pg_thread_pool_init(int nthreads) {
    if (g_init_done) return 0;
    // allow explicit init before once
    if (nthreads > 0) {
        // set env override for once
        char buf[16];
        snprintf(buf, sizeof(buf), "%d", nthreads);
        setenv("PICOGRAD_THREADS", buf, 1);
    }
    pthread_once(&g_once, pool_init_once);
    return 0;
}

void pg_thread_pool_fini(void) {
    if (!g_init_done) return;
    pg_pool_t *p = &g_pool;
    if (p->nthreads <= 1 || !p->threads) {
        g_init_done = 0;
        return;
    }
    pthread_mutex_lock(&p->mu);
    p->shutdown = 1;
    pthread_cond_broadcast(&p->cv);
    pthread_mutex_unlock(&p->mu);
    for (int i = 0; i < p->nthreads; i++) {
        pthread_join(p->threads[i], NULL);
    }
    free(p->threads);
    p->threads = NULL;
    pthread_mutex_destroy(&p->mu);
    pthread_cond_destroy(&p->cv);
    pthread_cond_destroy(&p->done_cv);
    g_init_done = 0;
}

int pg_thread_pool_size(void) {
    if (!g_init_done) {
        pthread_once(&g_once, pool_init_once);
    }
    return g_pool.nthreads;
}

void pg_parallel_for(size_t n, size_t threshold, pg_parallel_fn fn, void *ctx) {
    if (n == 0) return;
    if (!g_init_done) {
        pthread_once(&g_once, pool_init_once);
    }
    int nthreads = g_pool.nthreads;
    // serial fast path
    if (nthreads <= 1 || n < threshold) {
        fn(ctx, 0, n);
        return;
    }
    // split into chunks, at least threshold per chunk or nthreads chunks
    size_t chunk = (n + nthreads - 1) / nthreads;
    if (chunk < threshold) chunk = threshold;
    // count tasks
    size_t ntasks = (n + chunk - 1) / chunk;
    if (ntasks == 1) {
        fn(ctx, 0, n);
        return;
    }
    // Enqueue tasks
    pg_pool_t *p = &g_pool;
    pthread_mutex_lock(&p->mu);
    // enqueue all tasks
    for (size_t i = 0; i < ntasks; i++) {
        size_t start = i * chunk;
        size_t end = start + chunk;
        if (end > n) end = n;
        pg_task_t *task = malloc(sizeof(pg_task_t));
        if (!task) continue;
        task->fn = fn;
        task->ctx = ctx;
        task->start = start;
        task->end = end;
        task->next = NULL;
        if (p->tail) p->tail->next = task;
        else p->head = task;
        p->tail = task;
    }
    // wake workers
    pthread_cond_broadcast(&p->cv);
    // wait for all tasks done
    while (p->head || p->active) {
        pthread_cond_wait(&p->done_cv, &p->mu);
    }
    pthread_mutex_unlock(&p->mu);
}
