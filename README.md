conda create -n QIRO-LLM python=3.10
conda activate QIRO-LLM
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install beir==2.0.0
pip install mteb==1.1.1
pip install deepspeed==0.15.1
pip install peft==0.12.0
pip install transformers==4.44.2
pip install sentence-transformers==3.1.1
pip install datasets==2.21.0
pip install vllm==0.5.4
