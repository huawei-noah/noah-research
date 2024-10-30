from setuptools import setup


REQUIRED_PKGS = [
    "torch==1.12.0",
    "beautifulsoup4",
    "cryptography",
    "datasets",
    "decorator",
    "deepspeed==0.7.2",
    "dill==0.3.4",
    "dnspython==2.1.0",
    "filelock==3.7.1",
    "fire==0.5.0",
    "huggingface-hub",
    "jedi==0.17.0",
    "jieba==0.42.1",
    "Jinja2",
    "ninja==1.10.2.3",
    "nltk==3.8.1",
    "psutil==5.9.1",
    "pydantic==1.9.1",
    "PyYAML==6.0",
    "regex==2022.7.9",
    "requests",
    "responses==0.18.0",
    "scipy==1.10.1",
    "sentencepiece==0.1.96",
    "six==1.16.0",
    "soupsieve",
    "termcolor==2.4.0",
    "tokenizers==0.12.1",
    "tqdm",
    "traitlets",
    "transformers==4.19.0",
    "triton==1.0.0"
]


setup(
	name="text2code_mrpt",
	version="0.0.1",
	python_requires=">=3.8",
	author="Fenia Christopoulou",
	readme="README.md",
	packages=["source"],
	install_requires=REQUIRED_PKGS
)
