## Install torch first, no idea why i can't incorportate it into the requirements file, but don't mess with a working system
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
##Make sure submodules are present
git submodule update --init

pip install -r euler_req.txt