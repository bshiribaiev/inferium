#include "worker_team.hpp"

#include <algorithm>

namespace {

// Roughly a hundred microseconds of polling: long enough to bridge the serial work
// between matmuls, short enough that an idle server goes back to sleep.
constexpr int kSpinIterations = 1 << 16;

void pause_briefly() {
    __builtin_arm_yield();
}

}

WorkerTeam::WorkerTeam(int n_workers) {
    for (int i = 0; i < n_workers; ++i)
        workers_.emplace_back([this] { worker_loop(); });
}

WorkerTeam::~WorkerTeam() {
    stop_.store(true);
    {
        std::lock_guard<std::mutex> lock(mutex_);
    }
    wake_.notify_all();
    for (std::thread& worker : workers_)
        worker.join();
}

void WorkerTeam::run_job(int count, int chunk, void* context, RangeFn fn) {
    job_context_ = context;
    job_fn_ = fn;
    job_count_ = count;
    job_chunk_ = chunk;
    next_start_.store(0, std::memory_order_relaxed);
    workers_busy_.store((int)workers_.size(), std::memory_order_relaxed);

    // Publishing the generation and then checking for sleepers pairs with a worker
    // registering as a sleeper and then re-checking the generation: with both
    // sequentially consistent, at least one side sees the other, so no wake is lost.
    generation_.fetch_add(1, std::memory_order_seq_cst);
    if (sleepers_.load(std::memory_order_seq_cst) > 0) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
        }
        wake_.notify_all();
    }

    claim_chunks_until_done();
    while (workers_busy_.load(std::memory_order_acquire) > 0)
        pause_briefly();
}

void WorkerTeam::claim_chunks_until_done() {
    while (true) {
        int start = next_start_.fetch_add(job_chunk_, std::memory_order_relaxed);
        if (start >= job_count_) return;
        job_fn_(job_context_, start, std::min(start + job_chunk_, job_count_));
    }
}

bool WorkerTeam::wait_for_next_job(uint64_t& seen_generation) {
    for (int i = 0; i < kSpinIterations; ++i) {
        uint64_t current = generation_.load(std::memory_order_acquire);
        if (current != seen_generation) {
            seen_generation = current;
            return true;
        }
        if (stop_.load(std::memory_order_relaxed)) return false;
        pause_briefly();
    }

    std::unique_lock<std::mutex> lock(mutex_);
    sleepers_.fetch_add(1, std::memory_order_seq_cst);
    wake_.wait(lock, [&] {
        return stop_.load() ||
               generation_.load(std::memory_order_seq_cst) != seen_generation;
    });
    sleepers_.fetch_sub(1, std::memory_order_seq_cst);
    if (stop_.load()) return false;
    seen_generation = generation_.load(std::memory_order_acquire);
    return true;
}

void WorkerTeam::worker_loop() {
    uint64_t seen_generation = 0;
    while (wait_for_next_job(seen_generation)) {
        claim_chunks_until_done();
        workers_busy_.fetch_sub(1, std::memory_order_release);
    }
}
