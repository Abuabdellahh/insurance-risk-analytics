from setuptools import setup, find_packages

setup(
    name="insurance-risk-analytics",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="End-to-End Insurance Risk Analytics & Predictive Modeling",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/insurance-risk-analytics",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.1.0",
        "xgboost>=1.6.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "shap>=0.41.0",
        "dvc>=2.30.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "jupyter>=1.0.0",
        ],
    },
)
