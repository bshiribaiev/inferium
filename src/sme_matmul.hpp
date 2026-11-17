#pragma once

// Requires SME2. `packed_batch` holds the batch in groups of 16 rows, each group stored
// column by column like a weight panel.
void sme_multiply_blocks(const __fp16* panels, int cols, const float* packed_batch, int batch,
                         float* out, int out_dim, int block_start, int block_end);
