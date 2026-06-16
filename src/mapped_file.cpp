#include "mapped_file.hpp"

#include <stdexcept>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

int open_readonly(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error("open failed: " + path);
    }
    return fd;
}

size_t file_size(int fd) {
    struct stat st{};
    if (::fstat(fd, &st) != 0) {
        throw std::runtime_error("fstat failed");
    }
    if (st.st_size <= 0) {
        throw std::runtime_error("file is empty");
    }
    return static_cast<size_t>(st.st_size);
}

void* map_readonly(int fd, size_t size) {
    void* data = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) {
        throw std::runtime_error("mmap failed");
    }
    return data;
}

}  

MappedFile::MappedFile(const std::string& path) {
    int fd = open_readonly(path);
    try {
        size_ = file_size(fd);
        data_ = map_readonly(fd, size_);
    } catch (...) {
        ::close(fd);
        throw;
    }
    ::close(fd);
}

MappedFile::~MappedFile() {
    if (data_ != nullptr) {
        ::munmap(data_, size_);
    }
}
