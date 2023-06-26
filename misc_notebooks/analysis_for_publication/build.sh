docker build \
    --build-arg BASE_IMAGE="yfukai/conda-jax:latest-cuda" \
    -t yfukai/basicpy-benchmark:latest \
    "."
