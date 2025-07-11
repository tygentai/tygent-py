"""
Setup script for Tygent package.
"""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tygent",
    version="0.4.0",
    author="Tygent Team",
    author_email="info@tygent.ai",
    description="Transform LLM Agents into High-Performance Engines with DAG optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://tygent.ai",
    project_urls={
        "Bug Tracker": "https://github.com/tygent-ai/tygent/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "openai>=1.0.0",
        "aiohttp>=3.8",
    ],
)
