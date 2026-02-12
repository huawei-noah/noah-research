find . -name __pycache__ -type d -print -exec rm -rf {} \;
find . -name .ipynb_checkpoints -type d -print -exec rm -rf {} \;