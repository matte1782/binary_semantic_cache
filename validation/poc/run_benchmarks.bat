@echo off
REM Stage 1: Run Latency Benchmarks
REM Execute this from the poc folder

echo ============================================
echo Binary Semantic Cache - Latency Benchmarks
echo ============================================

cd /d "%~dp0"

echo.
echo Running benchmark_latency.py...
python benchmark_latency.py

echo.
echo ============================================
echo Benchmark complete. Check results folder.
echo ============================================
pause

