This is the pytorch implementation of our paper [Video-RAG: Visually-aligned Retrieval-Augmented Long Video Comprehension](https://arxiv.org/abs/2411.13093).

This repo is built upon LLaVA-NeXT:

- Step 1: Clone and build LLaVA-NeXT conda environment:

```
git clone https://github.com/LLaVA-VL/LLaVA-NeXT
cd LLaVA-NeXT
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # Enable PEP 660 support.
pip install -e ".[train]"
```
Then install the following packages in llava environment:
```
pip install spacy faiss-cpu easyocr ffmpeg-python
pip install torch==2.1.2 torchaudio
python -m spacy download en_core_web_sm
# Optional: pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.0.0/en_core_web_sm-3.0.0.tar.gz
```

- Step 2: Clone and build another conda environment for APE by: 

```
git clone https://github.com/shenyunhang/APE
cd APE
pip3 install -r requirements.txt
python3 -m pip install -e .
```

- Step 3: Copy all the files in 'vidrag_pipeline' under the root dir of LLaVA-NeXT;

- Step 4: Copy all the files in 'ape_tools' under the 'demo' dir of APE;

- Step 5: Opening a service of APE by running the code under APE/demo:

```
python demo/ape_service.py
```

- Step 6: You can now run our pipeline build upon LLaVA-Video-7B by:

```
python vidrag_pipeline.py
```

- Note that you can also use our pipeline in any LVLMs, only need to modify #line 160 and #line 175 in vidrag_pipeline.py

If you find our paper and code useful in your research, please consider giving a star and citation:

```
@misc{luo2024videoragvisuallyalignedretrievalaugmentedlong,
      title={Video-RAG: Visually-aligned Retrieval-Augmented Long Video Comprehension}, 
      author={Yongdong Luo and Xiawu Zheng and Xiao Yang and Guilin Li and Haojia Lin and Jinfa Huang and Jiayi Ji and Fei Chao and Jiebo Luo and Rongrong Ji},
      year={2024},
      eprint={2411.13093},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.13093}, 
}
```
