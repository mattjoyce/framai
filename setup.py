"""
Setup file for FRAMAI package installation
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README if it exists
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    long_description = readme_file.read_text()

setup(
    name="framai",
    version="1.0.0",
    author="Matt Joyce",
    description="Field Recording Audio/Media Analysis & Integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mattjoyce/framai",  # Update if you have a repo
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=[
        # Core
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "PyYAML>=6.0",

        # CLI & Console
        "click>=8.0.0",
        "rich>=13.0.0",

        # API clients
        "openai>=1.0.0",
        "openmeteo-requests>=1.0.0",
        "requests>=2.31.0",
        "requests-cache>=1.0.0",
        "retry-requests>=2.0.0",

        # Image processing
        "Pillow>=10.0.0",

        # Audio processing (optional - may not work on Python 3.13+)
        "pydub; python_version < '3.13'",
        "openai-whisper; python_version < '3.13'",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "audio": [
            # Full audio support for older Python versions
            "pydub",
            "openai-whisper",
            "ffmpeg-python",
        ],
    },
    entry_points={
        "console_scripts": [
            "fram-cli=fram_cli:cli",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml"],
    },
    zip_safe=False,
)
