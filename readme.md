This is the pytorch implementation of our paper [Video-RAG: Visually-aligned Retrieval-Augmented Long Video Comprehension](https://arxiv.org/abs/2411.13093).

This repo is built upon LLaVA-Next:

- Step 1: Clone and build LLaVA-NeXT conda environment, then install the following packages in llava environment:

```
git clone https://github.com/LLaVA-VL/LLaVA-NeXT
pip install spacy faiss-cpu easyocr ffmpeg-python
pip install torch==2.1.2 torchaudio
python -m spacy download en_core_web_sm
# Optional: pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.0.0/en_core_web_sm-3.0.0.tar.gz
```

- Step 2: Clone and build APE conda environment following: 

```
https://github.com/shenyunhang/APE
```

- Step 3: Copy the files in 'vidrag_pipeline' under the root dir of LLaVA-NeXT;

- Step 4: Copy the files in 'ape_tools' under the 'demo' dir of APE;

- Step 5: Opening a service of APE by running the code under APE:

```
python ape_service.py
```

- Step 6: You can now run our pipeline build upon LLaVA-Video-7B by:

```
python vidrag_pipeline.py
```

- Note that you can also use our pipeline in any LVLMs.

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
