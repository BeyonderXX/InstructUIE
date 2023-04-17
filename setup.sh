pip install -r requirements.txt
apt-get -y install ninja-build
apt-get -y install pdsh
python -c "import nltk; nltk.download('punkt', quiet=True)"
conda install -y mpi4py