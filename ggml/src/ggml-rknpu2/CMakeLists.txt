# The RKNPU2 backend uses NEON intrinsics and links the vendor aarch64 librknnrt.so,
# so it can only be built for aarch64 (natively or with a cross toolchain).
if (NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    message(FATAL_ERROR "GGML_RKNPU2 requires an aarch64 target (got ${CMAKE_SYSTEM_PROCESSOR}); "
                        "cross-compile with an aarch64 toolchain, see ggml/src/ggml-rknpu2/README.md")
endif()

# Finding OpenMP
find_package(OpenMP REQUIRED)

# Adding backend files
ggml_add_backend_library(ggml-rknpu2
    ggml-rknpu2.cpp
    rknpu2-quantization.cpp
    rknpu2-calibration.cpp
    rknpu2-configuration.cpp
)

# Linking rknpu2 header files
target_include_directories(ggml-rknpu2
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/libs/include
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Linking rknpu2 library
target_link_libraries(ggml-rknpu2 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/librknnrt.so
)

target_link_options(ggml-rknpu2 PUBLIC
    "-Wl,-rpath-link,${CMAKE_CURRENT_SOURCE_DIR}/libs"
)

# Adding OpenMP in build
if(OpenMP_FOUND)
    add_definitions(-DHAVE_OPENMP=1)
    target_compile_options(ggml-rknpu2 PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(ggml-rknpu2 PRIVATE ${OpenMP_CXX_FLAGS})
endif()
