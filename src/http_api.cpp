#include "http_api.hpp"

#include <iostream>

#include "httplib.h"
#include "json.hpp"

namespace {

constexpr int kDefaultMaxTokens = 64;
constexpr int kMaxAllowedTokens = 2048;

void reply_error(httplib::Response& response, int status, const std::string& message) {
    nlohmann::json body{{"error", message}};
    response.status = status;
    response.set_content(body.dump(), "application/json");
}

}

void run_http_api(const std::string& host, int port, GenerateFn generate) {
    httplib::Server server;

    server.Get("/health", [](const httplib::Request&, httplib::Response& response) {
        response.set_content(nlohmann::json{{"status", "ok"}}.dump(), "application/json");
    });

    server.Post("/generate", [&generate](const httplib::Request& request,
                                         httplib::Response& response) {
        nlohmann::json payload = nlohmann::json::parse(request.body, nullptr, false);
        if (payload.is_discarded() || !payload.contains("prompt") ||
            !payload["prompt"].is_string()) {
            reply_error(response, 400, "body must be JSON with a string \"prompt\"");
            return;
        }

        int max_tokens = payload.value("max_tokens", kDefaultMaxTokens);
        if (max_tokens <= 0 || max_tokens > kMaxAllowedTokens) {
            reply_error(response, 400, "max_tokens must be in [1, 2048]");
            return;
        }

        try {
            GenerationResult result = generate(payload["prompt"].get<std::string>(), max_tokens);
            nlohmann::json body{{"text", result.text},
                                {"prompt_tokens", result.prompt_tokens},
                                {"generated_tokens", result.generated_tokens}};
            response.set_content(body.dump(), "application/json");
        } catch (const std::exception& error) {
            reply_error(response, 503, error.what());
        }
    });

    std::cout << "listening on http://" << host << ":" << port << "\n" << std::flush;
    if (!server.listen(host, port)) {
        throw std::runtime_error("failed to bind " + host + ":" + std::to_string(port));
    }
}
