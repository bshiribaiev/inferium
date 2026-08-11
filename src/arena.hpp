#pragma once
#include <cstddef>
#include <memory>

class Arena {
    public:
        explicit Arena(size_t capacity_bytes);

        float* alloc_floats(size_t count);

        void reset() {
            used_ = 0;
        }

        void rewind(size_t mark) {
            used_ = mark;
        }

        size_t used() const {
            return used_;
        }

        size_t capacity() const {
            return capacity_;
        }

        Arena(const Arena&) = delete;
        Arena& operator= (const Arena&) = delete;

    private:
        static constexpr size_t kAlignment = 64;

        struct AlignedDelete {
            void operator() (std::byte* p) const {
                ::operator delete(p, std::align_val_t(kAlignment));
            }
        };

        std::unique_ptr<std::byte[], AlignedDelete> buffer_;
        size_t capacity_;
        size_t used_ = 0;
};

class ArenaScope {
    public:
        explicit ArenaScope(Arena& arena)
            : arena_(arena), mark_(arena.used()) {}

        ~ArenaScope() {
            arena_.rewind(mark_);
        }

        ArenaScope(const ArenaScope&) = delete;
        ArenaScope& operator= (const ArenaScope&) = delete;

    private:
        Arena& arena_;
        size_t mark_;
};
