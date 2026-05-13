# ============================================================
# Single stage – build + runtime
# Base: CUDA 12.8.2 devel + Ubuntu 24.04
# ============================================================
FROM nvidia/cuda:12.8.2-devel-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
        libcurl4-openssl-dev \
        libgomp1 \
        ninja-build \
        pkg-config \
        python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/uninference
COPY . .

ARG CUDA_ARCH=89

RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so \
           /usr/local/cuda/lib64/stubs/libcuda.so.1

ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs

RUN cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=ON \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
        -DCMAKE_CUDA_FLAGS="-Xcudafe --diag_suppress=177" \
        -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
        -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
        -DGGML_AVX2=ON \
        -DGGML_FMA=ON \
        -DGGML_F16C=ON \
        -DGGML_CUDA_FA_ALL_QUANTS=ON \
    && cmake --build build --parallel "$(nproc)" \
    && cp build/bin/llama-server /usr/local/bin/llama-server \
    && cp build/bin/llama-bench  /usr/local/bin/llama-bench \
    && cp build/bin/llama-cli    /usr/local/bin/llama-cli \
    && find build -name "*.so*" ! -path "*/CMakeFiles/*" -exec cp -P {} /usr/local/lib/ \; \
    && ldconfig \
    && rm -rf /opt/uninference

EXPOSE 8080
ENTRYPOINT ["llama-server"]
