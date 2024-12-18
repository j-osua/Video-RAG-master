# Video-RAG: Visually-aligned Retrieval-Augmented Long Video Comprehension

[![Arxiv](https://img.shields.io/badge/Arxiv-2411.13093-red)](https://arxiv.org/abs/2411.13093)
![](https://img.shields.io/badge/Task-VideoQA-blue) [![Arxiv](https://img.shields.io/badge/Web-Project_Page-yellow)](https://video-rag.github.io/)

<font size=7><div align='center' > [[🍎 Project Page](https://video-rag.github.io/)] [[📖 arXiv Paper](https://arxiv.org/pdf/2411.13093)] </div></font>

## 😮 Highlights
<div style="text-align:center">
  <img src="https://github.com/user-attachments/assets/939f31a8-3e61-4b96-9d43-21c6b278628c" width="100%">
</div>

- **We integrate RAG into open-source LVLMs:** Video-RAG incorporates three types of visually-aligned auxiliary texts (OCR, ASR, and object detection) processed by external tools and retrieved via RAG, enhancing the LVLM. It’s implemented using completely open-source tools, without the need for any commercial APIs.
- **We design a versatile plug-and-play RAG-based pipeline for any LVLM:** Video-RAG offers a training-free solution for a wide range of LVLMs, delivering performance improvements with minimal additional resource requirements.
- **We achieve proprietary-level performance with open-source models:** Applying Video-RAG to a 72B open-source model yields state-of-the-art performance in Video-MME, surpassing models such as Gemini-1.5-Pro.
![framework](https://github.com/user-attachments/assets/9c9b176c-10a8-483e-be6b-de72b2b68191)
![results](https://github.com/user-attachments/assets/fba0cb38-af03-4574-8826-8664ceb7ffd8)

## 📈 Ranking
- **Rank #3 on [Video-MME (with subs)](https://paperswithcode.com/sota/zero-shot-video-question-answer-on-video-mme-1)**
- **Rank #3 on [EgoSchma (fullset)](https://paperswithcode.com/sota/zero-shot-video-question-answer-on-egoschema-1)**

## 🔨 Usage

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
pip install torch==2.1.2 torchaudio numpy
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

## ✏️ Citation

If you find our paper and code useful in your research, please consider giving a star ⭐ and citation 📝:

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
