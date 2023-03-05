pip install -r requirements.txt
apt-get -y install ninja-build
apt-get -y install pdsh
python -c "import nltk; nltk.download('punkt', quiet=True)"
conda install -y mpi4py
# wget https://github.com/ninja-build/ninja/releases/download/v1.10.2/ninja-linux.zip
# sudo unzip ninja-linux.zip -d /usr/local/bin/