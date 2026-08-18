import setuptools

with open("requirements.txt", 'r', encoding='utf-8') as f:
    requirements = f.read().strip().splitlines()

setuptools.setup(
    name="UltraEval",
    version="0.1",
    author="UltraEval Team",
    author_email="",
    description="An open source framework for evaluating foundation models",
    packages=setuptools.find_packages(),
    python_requires=">=3.10",
    install_requires=requirements,
)
