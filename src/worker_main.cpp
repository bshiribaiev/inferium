#include <iostream>
#include <memory>
#include <string>

#include <grpcpp/grpcpp.h>

#include "engine.hpp"
#include "forward_pass.hpp"
#include "inference.grpc.pb.h"
#include "model_loader.hpp"
#include "server_options.hpp"

namespace {

class InferenceServiceImpl final : public inferium::Inference::Service {
    public:
        explicit InferenceServiceImpl(InferenceEngine& engine) : engine_(engine) {}

        grpc::Status Generate(grpc::ServerContext*,
                              const inferium::GenerateRequest* request,
                              inferium::GenerateReply* reply) override {
            try {
                GenerationResult result =
                    engine_.submit(request->prompt(), request->max_tokens()).get();
                reply->set_text(result.text);
                reply->set_prompt_tokens(result.prompt_tokens);
                reply->set_generated_tokens(result.generated_tokens);
                return grpc::Status::OK;
            } catch (const std::exception& error) {
                return grpc::Status(grpc::StatusCode::INTERNAL, error.what());
            }
        }

        grpc::Status Health(grpc::ServerContext*, const inferium::HealthRequest*,
                            inferium::HealthReply* reply) override {
            reply->set_ok(true);
            return grpc::Status::OK;
        }

    private:
        InferenceEngine& engine_;
};

}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: inferium_worker <model.gguf> [--port N] [--batch N]"
                     " [--max-seq N] [--threads N]\n";
        return 1;
    }

    ServerOptions options = parse_server_options(argc, argv);
    if (options.threads > 0) set_num_threads(options.threads);

    LoadedModel loaded(argv[1]);

    EngineConfig config;
    config.max_batch = options.max_batch;
    config.max_seq = options.max_seq;
    InferenceEngine engine(loaded.model(), loaded.tokenizer(), config);

    InferenceServiceImpl service(engine);
    std::string address = options.host + ":" + std::to_string(options.port);

    grpc::ServerBuilder builder;
    builder.AddListeningPort(address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
    if (!server) {
        std::cerr << "failed to bind " << address << "\n";
        return 1;
    }

    std::cout << "worker listening on " << address << "\n" << std::flush;
    server->Wait();
    return 0;
}
