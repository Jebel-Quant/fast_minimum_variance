#!/bin/bash -eu
# ClusterFuzzLite build script — installs fast_minimum_variance and compiles each Python
# harness in tests/fuzz/ via OSS-Fuzz's compile_python_fuzzer helper.

cd "$SRC"

# Pin pip so the build environment is reproducible and only changes through a
# reviewed bump (the same rationale as the SHA-pinned base image).
pip3 install --upgrade "pip==24.3.1"

# Install the package and its runtime dependencies so PyInstaller can discover
# and bundle fast_minimum_variance into each frozen fuzzer binary.
pip3 install .

# PyInstaller does not discover numpy's C-extension submodules on its own, so
# the frozen fuzzer crashes at runtime with
# "No module named 'numpy._core._exceptions'". --collect-all pulls in every
# numpy submodule, data file and shared library.
# nullglob so an empty tests/fuzz/ yields zero iterations rather than passing
# the literal unexpanded pattern to compile_python_fuzzer; fail loudly instead
# of silently building nothing.
shopt -s nullglob
fuzzers=(tests/fuzz/fuzz_*.py)
if [ ${#fuzzers[@]} -eq 0 ]; then
  echo "ERROR: no fuzz harnesses found in tests/fuzz/ (expected fuzz_*.py)" >&2
  exit 1
fi
for fuzzer in "${fuzzers[@]}"; do
  compile_python_fuzzer "$fuzzer" --collect-all numpy --collect-all scipy --collect-all cvxpy --collect-all osqp
done
