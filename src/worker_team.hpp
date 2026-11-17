#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

// Runs one range job at a time across the calling thread plus its workers. Only one
// thread may call run() at once; the engine's single scheduler thread guarantees that.
class WorkerTeam {
    public:
        explicit WorkerTeam(int n_workers);
        ~WorkerTeam();

        template <typename Fn>
        void run(int count, int chunk, Fn& process_range) {
            run_job(count, chunk, &process_range, [](void* context, int start, int end) {
                (*static_cast<Fn*>(context))(start, end);
            });
        }

        int thread_count() const {
            return (int)workers_.size() + 1;
        }

        WorkerTeam(const WorkerTeam&) = delete;
        WorkerTeam& operator= (const WorkerTeam&) = delete;

    private:
        using RangeFn = void (*)(void*, int, int);

        void run_job(int count, int chunk, void* context, RangeFn fn);
        void claim_chunks_until_done();
        bool wait_for_next_job(uint64_t& seen_generation);
        void worker_loop();

        std::vector<std::thread> workers_;
        void* job_context_ = nullptr;
        RangeFn job_fn_ = nullptr;
        int job_count_ = 0;
        int job_chunk_ = 1;

        // Each hot counter gets its own 128-byte cache line so threads hammering one
        // don't keep invalidating the line that holds another.
        alignas(128) std::atomic<uint64_t> generation_{0};
        alignas(128) std::atomic<int> next_start_{0};
        alignas(128) std::atomic<int> workers_busy_{0};
        alignas(128) std::atomic<int> sleepers_{0};
        std::atomic<bool> stop_{false};
        std::mutex mutex_;
        std::condition_variable wake_;
};
