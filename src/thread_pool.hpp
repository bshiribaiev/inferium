#pragma once
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

class ThreadPool {
    private:
        std::vector<std::thread> workers_;
        std::queue<std::function<void()>> tasks_;
        std::mutex mutex_;
        std::condition_variable condition_;
        bool stop_ = false;

        void worker_loop();

    public:
        explicit ThreadPool(unsigned num_workers);

        ~ThreadPool();

        void submit(std::function<void()> task);

        // No copying - workers and synchronization primitives are not copyable
        ThreadPool(const ThreadPool&) = delete;
        ThreadPool& operator= (const ThreadPool&) = delete;
};
