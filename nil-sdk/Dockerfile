FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir pytest

RUN rm -rf benchmark_store benchmark_results

# Default: full benchmark (takes ~2-5 min with StubProvider)
# Override with --smoke for CI:  docker run --rm nil-benchmark python run_benchmarks.py --smoke
CMD ["python", "scripts/run_benchmarks.py"]
