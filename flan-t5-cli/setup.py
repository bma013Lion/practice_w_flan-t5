from setuptools import setup, find_packages

setup(
    name="simple-inference-flan-t5",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "rich>=13.0.0",
    ],
    entry_points={
        "console_scripts": [
            "flan-t5-cli=flan_t5_cli.cli:main",  # This should match your command
        ],
    },
    python_requires=">=3.8",
)