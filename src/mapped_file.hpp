#pragma once
#include <cstddef>
#include <string>

class MappedFile {
    private:
        void* data_ = nullptr;
        size_t size_ = 0;

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
