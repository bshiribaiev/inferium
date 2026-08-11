#include "worker_client.hpp"

#include <chrono>
#include <stdexcept>

#include <grpcpp/grpcpp.h>

namespace {

std::chrono::system_clock::time_point deadline_from_now(int timeout_ms) {
    return std::chrono::system_clock::now() + std::chrono::milliseconds(timeout_ms);
}

}

WorkerPool::WorkerPool(const std::vector<std::string>& addresses, int timeout_ms,
                       int max_retries)
    : timeout_ms_(timeout_ms), max_retries_(max_retries) {
    if (addresses.empty()) {
        throw std::runtime_error("coordinator needs at least one --workers address");
    }
    for (const std::string& address : addresses) {
        Worker worker;
        worker.address = address;
        worker.stub = inferium::Inference::NewStub(
            grpc::CreateChannel(address, grpc::InsecureChannelCredentials()));
        workers_.push_back(std::move(worker));
    }
}

WorkerPool::Worker& WorkerPool::next_worker() {
    size_t index = cursor_.fetch_add(1) % workers_.size();
    return workers_[index];
}

GenerationResult WorkerPool::generate(const std::string& prompt, int max_tokens) {
    inferium::GenerateRequest request;
    request.set_prompt(prompt);
    request.set_max_tokens(max_tokens);

    std::string last_error;
    // Each retry advances the round-robin cursor, so a failed worker is skipped
    // rather than retried into.
    for (int attempt = 0; attempt <= max_retries_; ++attempt) {
        Worker& worker = next_worker();

        grpc::ClientContext context;
        context.set_deadline(deadline_from_now(timeout_ms_));
        inferium::GenerateReply reply;

        grpc::Status status = worker.stub->Generate(&context, request, &reply);
        if (status.ok()) {
            GenerationResult result;
            result.text = reply.text();
            result.prompt_tokens = reply.prompt_tokens();
            result.generated_tokens = reply.generated_tokens();
            return result;
        }
        last_error = worker.address + ": " + status.error_message();
    }

    throw std::runtime_error("all workers failed, last error - " + last_error);
}

std::vector<std::string> WorkerPool::healthy_workers() {
    std::vector<std::string> healthy;
    for (Worker& worker : workers_) {
        grpc::ClientContext context;
        context.set_deadline(deadline_from_now(2000));
        inferium::HealthRequest request;
        inferium::HealthReply reply;
        if (worker.stub->Health(&context, request, &reply).ok() && reply.ok()) {
            healthy.push_back(worker.address);
        }
    }
    return healthy;
}
