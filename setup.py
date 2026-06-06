from setuptools import setup, find_packages

setup(
    name="epistemic-drift-eval",
    version="0.1.0",
    description="Framework for measuring conversational epistemic drift in aligned LLMs",
    author="Nicolás Torres",
    author_email="nicolas.torresr@usm.cl",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "openai>=1.30.0",
        "anthropic>=0.28.0",
        "sentence-transformers>=3.0.0",
        "datasets>=2.19.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "pyyaml>=6.0.1",
        "tqdm>=4.66.0",
        "rich>=13.7.0",
        "jsonlines>=4.0.0",
        "python-dotenv>=1.0.0",
        "tenacity>=8.3.0",
    ],
    extras_require={
        "dev": ["pytest", "pytest-asyncio", "jupyter"],
        "hf": ["transformers>=4.40.0", "torch>=2.2.0"],
    },
)
