from setuptools import setup, find_packages

setup(
    name="reasoning_agent",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "requests",
        "textblob",
        "beautifulsoup4",
        "numpy",
        "pandas"
    ],
    description="An advanced reasoning agent powered by any LLM model for decision-making.",
    author="Ashish Sharma",
    author_email="ashishrsharma99@gmail.com",
    url="https://github.com/Ashishsharma-12/Reasoning-Agent",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
