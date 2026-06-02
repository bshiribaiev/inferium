#include <iostream>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "gguf_parser.hpp"

class MappedFile {
    private:
        void* data_ = nullptr;
        size_t size_ = 0;
        int fd_ = -1;

    public:
        explicit MappedFile(const std::string& path);

        ~MappedFile();

        const void* data() const {
            return data_;
        }

        size_t size() const {
            return size_;
        }
        
        // Prevent copy - objects must not own same fd
        MappedFile(const MappedFile&) = delete;
        MappedFile& operator= (const MappedFile&) = delete;
};

MappedFile::MappedFile(const std::string& path) {
    fd_ = ::open(path.c_str(), O_RDONLY);

    if (fd_ < 0) {
        throw std::runtime_error("open failed: " + path);
    }

    struct stat st{};
    if (::fstat(fd_, &st) != 0) {
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("fstat failed");
    }

    if (st.st_size <= 0) {
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("file is empty");
    }

    size_ = static_cast<size_t>(st.st_size);
    data_ = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (data_ == MAP_FAILED) {
        data_ = nullptr;
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("mmap failed");
    }
}

MappedFile::~MappedFile() {
    if (data_ != nullptr) {
        ::munmap(data_, size_);
    }

    if (fd_ >= 0) { 
        ::close(fd_);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        return 1;
    }

    MappedFile model(argv[1]);
    const uint8_t* base = static_cast<const uint8_t*>(model.data());

    GgufParser parser(base, model.size());
    std::cout << "arch: "             << parser.arch             << "\n";
    std::cout << "block_count: "      << parser.block_count      << "\n";
    std::cout << "embedding_length: " << parser.embedding_length << "\n";
    std::cout << "tensors: "          << parser.tensors.size()   << "\n";
    std::cout << "offset: " << parser.tensor_data_offset;    
    return 0;
}