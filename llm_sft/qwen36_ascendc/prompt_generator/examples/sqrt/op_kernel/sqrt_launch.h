#ifndef SQRT_LAUNCH_H
#define SQRT_LAUNCH_H

#include <cstdint>
#include <tuple>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

std::tuple<int64_t, int64_t, int64_t> calc_sqrt_tiling_params(int64_t totalLength);

extern "C" {
void launch_sqrt_kernel_float   (GM_ADDR x, GM_ADDR z, int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileSize, void* stream);
void launch_sqrt_kernel_half    (GM_ADDR x, GM_ADDR z, int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileSize, void* stream);
void launch_sqrt_kernel_bfloat16(GM_ADDR x, GM_ADDR z, int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileSize, void* stream);
}

#endif // SQRT_LAUNCH_H
