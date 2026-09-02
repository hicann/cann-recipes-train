set(SQRT_KERNEL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/sqrt_kernel.cpp
)

set(SQRT_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/sqrt_plugin.cpp
)

register_direct_launch_op(
    "${SQRT_KERNEL_SRCS}"
    op_kernel
    "${SQRT_PLUGIN_SRCS}"
    op_kernel
    "--npu-arch=dav-2201"
)
