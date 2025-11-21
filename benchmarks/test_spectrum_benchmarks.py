"""
Benchmarks for spectrum_utils performance testing.

This module contains benchmarks for the most performance-critical operations
in spectrum_utils, particularly focusing on Numba JIT-compiled functions
vs regular Python implementations.
"""

import numpy as np
import pytest

from spectrum_utils.spectrum import MsmsSpectrum, MsmsSpectrumJit


@pytest.fixture(scope="session")
def sample_data():
    """Generate sample spectrum data for benchmarking."""
    np.random.seed(42)  # Reproducible results
    n_peaks = 1000
    mz = np.sort(np.random.uniform(100, 2000, n_peaks))
    intensity = np.random.exponential(1000, n_peaks)
    return {
        "identifier": "benchmark_spectrum",
        "precursor_mz": 500.0,
        "precursor_charge": 2,
        "mz": mz,
        "intensity": intensity,
        "retention_time": 60.0,
    }


@pytest.fixture(scope="session")
def large_sample_data():
    """Generate large spectrum data for stress testing."""
    np.random.seed(42)
    n_peaks = 10000  # 10x larger for stress testing
    mz = np.sort(np.random.uniform(100, 2000, n_peaks))
    intensity = np.random.exponential(1000, n_peaks)
    return {
        "identifier": "large_benchmark_spectrum",
        "precursor_mz": 500.0,
        "precursor_charge": 2,
        "mz": mz,
        "intensity": intensity,
        "retention_time": 60.0,
    }


@pytest.fixture(scope="session")
def comparison_data():
    """Generate data for performance comparison."""
    np.random.seed(42)
    sizes = [100, 1000, 5000, 10000]
    datasets = {}

    for size in sizes:
        mz = np.sort(np.random.uniform(100, 2000, size))
        intensity = np.random.uniform(0, 1, size)
        datasets[f"size_{size}"] = {
            "identifier": f"comparison_spectrum_{size}",
            "precursor_mz": 500.0,
            "precursor_charge": 2,
            "mz": mz,
            "intensity": intensity,
            "retention_time": 60.0,
        }
    return datasets


class TestSpectrumPerformance:
    """Benchmarks for spectrum processing operations."""

    def test_spectrum_creation_regular(self, benchmark, sample_data):
        """Benchmark regular MsmsSpectrum creation."""
        result = benchmark.pedantic(
            MsmsSpectrum, kwargs=sample_data, rounds=50, iterations=1
        )
        assert len(result.mz) == len(sample_data["mz"])

    def test_spectrum_creation_jit(self, benchmark, sample_data):
        """Benchmark JIT MsmsSpectrum creation."""
        _ = MsmsSpectrumJit(**sample_data)

        result = benchmark.pedantic(
            MsmsSpectrumJit, kwargs=sample_data, rounds=50, iterations=1
        )
        assert len(result.mz) == len(sample_data["mz"])

    def test_spectrum_creation_large_regular(
        self, benchmark, large_sample_data
    ):
        """Benchmark regular MsmsSpectrum creation with large data."""
        result = benchmark.pedantic(
            MsmsSpectrum, kwargs=large_sample_data, rounds=50, iterations=1
        )
        assert len(result.mz) == len(large_sample_data["mz"])

    def test_spectrum_creation_large_jit(self, benchmark, large_sample_data):
        """Benchmark JIT MsmsSpectrum creation with large data."""
        _ = MsmsSpectrumJit(**large_sample_data)

        result = benchmark.pedantic(
            MsmsSpectrumJit, kwargs=large_sample_data, rounds=50, iterations=1
        )
        assert len(result.mz) == len(large_sample_data["mz"])

    def test_spectrum_round_regular(self, benchmark, sample_data):
        """Benchmark rounding operation on regular spectrum."""

        def setup():
            return (MsmsSpectrum(**sample_data),), {}

        result = benchmark.pedantic(
            lambda s: s.round(decimals=2), setup=setup, rounds=50, iterations=1
        )
        assert result is not None

    def test_spectrum_round_jit(self, benchmark, sample_data):
        """Benchmark rounding operation on JIT spectrum."""
        _ = MsmsSpectrumJit(**sample_data).round(2)

        def setup():
            return (MsmsSpectrumJit(**sample_data),), {}

        result = benchmark.pedantic(
            lambda s: s.round(2), setup=setup, rounds=50, iterations=1
        )
        assert result is not None

    def test_spectrum_filter_intensity_regular(self, benchmark, sample_data):
        """Benchmark intensity filtering on regular spectrum."""

        def setup():
            return (MsmsSpectrum(**sample_data),), {}

        result = benchmark.pedantic(
            lambda s: s.filter_intensity(min_intensity=0.01),
            setup=setup,
            rounds=50,
            iterations=1,
        )
        assert result is not None

    def test_spectrum_filter_intensity_jit(self, benchmark, sample_data):
        """Benchmark intensity filtering on JIT spectrum."""
        _ = MsmsSpectrumJit(**sample_data).filter_intensity(0.01)

        def setup():
            return (MsmsSpectrumJit(**sample_data),), {}

        result = benchmark.pedantic(
            lambda s: s.filter_intensity(0.01),
            setup=setup,
            rounds=50,
            iterations=1,
        )
        assert result is not None

    def test_spectrum_scale_intensity_regular(self, benchmark, sample_data):
        """Benchmark intensity scaling on regular spectrum."""

        def setup():
            return (MsmsSpectrum(**sample_data),), {}

        result = benchmark.pedantic(
            lambda s: s.scale_intensity(scaling="root"),
            setup=setup,
            rounds=50,
            iterations=1,
        )
        assert result is not None

    def test_spectrum_scale_intensity_jit(self, benchmark, sample_data):
        """Benchmark intensity scaling on JIT spectrum."""
        _ = MsmsSpectrumJit(**sample_data).scale_intensity("root")

        def setup():
            return (MsmsSpectrumJit(**sample_data),), {}

        result = benchmark.pedantic(
            lambda s: s.scale_intensity("root"),
            setup=setup,
            rounds=50,
            iterations=1,
        )
        assert result is not None


class TestSpectrumComparison:
    """Benchmarks comparing JIT vs regular implementations."""

    @pytest.mark.parametrize("size", [100, 1000, 5000, 10000])
    def test_creation_performance_comparison(
        self, benchmark, comparison_data, size
    ):
        """Compare creation performance across different spectrum sizes."""
        data = comparison_data[f"size_{size}"]

        benchmark.extra_info["spectrum_size"] = size

        regular_result = benchmark.pedantic(
            MsmsSpectrum, kwargs=data, rounds=50, iterations=1
        )

        assert len(regular_result.mz) == size

    @pytest.mark.parametrize("size", [100, 1000, 5000, 10000])
    def test_jit_creation_performance_comparison(
        self, benchmark, comparison_data, size
    ):
        """Compare JIT creation performance across different spectrum sizes."""
        data = comparison_data[f"size_{size}"]

        benchmark.extra_info["spectrum_size"] = size
        benchmark.extra_info["implementation"] = "jit"

        _ = MsmsSpectrumJit(**data)

        jit_result = benchmark.pedantic(
            MsmsSpectrumJit, kwargs=data, rounds=50, iterations=1
        )

        assert len(jit_result.mz) == size
