# Demo of CLIP for Conditioned Image Retrieval on CIRR dataset

## About The Project

This demo is a part of the final project for the CS336 course - Multimedia Information Retrieval at UIT.

In this demo, we present an interactive system based on a combiner network, trained using contrastive learning, that combines visual and textual features obtained from the OpenAI CLIP network. The system uses Faiss library for efficient similarity search.

### Built With
* [Python](https://www.python.org/)
* [PyTorch](https://pytorch.org/)
* [Torchvision](https://pytorch.org/vision/stable/index.html)
* [CLIP](https://github.com/openai/CLIP)
* [Flask](https://flask.palletsprojects.com/en/2.0.x/)
* [Bootstrap](https://getbootstrap.com/)
* [Faiss](https://github.com/facebookresearch/faiss)

### Tested On
* Windows 11
* Python 3.11.7

## Getting Started

To get a local copy up and running follow these simple steps.

### Installation
 
1. Clone the repo
```sh
git clone https://github.com/nganngants/CIR-demo
```
2. Install Python dependencies
```sh
pip install -r requirements.txt
```
3. Download [**CIRR**](https://cuberick-orion.github.io/CIRR/) datasets. 
## Usage
Here's a brief description of each and every file and folder in the repo:

* ```utils.py```: Utils file
* ```model.py```: Combiner model definition file
* ```data_utils.py```: Dataset loading and preprocessing utils file
* ```extract_features.py```: Feature extraction file
* ```hubconf.py```: Torch Hub config file
* ```app.py```: Flask server file
* ```static```: Flask static files folder
* ```templates```: Flask templates folder

### Data Preparation
To properly work with the codebase [**CIRR**](https://cuberick-orion.github.io/CIRR/) datasets should have the following structure:

```
project_base_path
└───  cirr_dataset       
       └─── dev
            | dev-0-0-img0.png
            | dev-0-0-img1.png
            | dev-0-1-img0.png
            | ...
       
       └─── test1
            | test1-0-0-img0.png
            | test1-0-0-img1.png
            | test1-0-1-img0.png 
            | ...
       
       └─── cirr
            └─── captions
                | cap.rc2.test1.json
                | cap.rc2.train.json
                | cap.rc2.val.json
                
            └─── image_splits
                | split.rc2.test1.json
                | split.rc2.train.json
                | split.rc2.val.json
```

### Feature Extraction
Before launching the demo it is necessary to extract the features 
using the following command
```shell
python extract_features.py
```

### Run the Demo
Start the server and run the demo using the following command
```shell
python app.py
```
By default, the server run on port 5000 of localhost address: http://127.0.0.1:5000/

## Original repository
This project is based on the [official repository](https://github.com/ABaldrati/CLIP4CirDemo) of the paper [Effective conditioned and composed image retrieval combining CLIP-based features](https://openaccess.thecvf.com/content/CVPR2022/papers/Baldrati_Effective_Conditioned_and_Composed_Image_Retrieval_Combining_CLIP-Based_Features_CVPR_2022_paper.pdf)

## Acknowledgement
Our demo is based on the respective official repositories, we thank the authors to release their code. If you use the related parts, please cite the corresponding papers.

## Reference
[DEMO paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Baldrati_Effective_Conditioned_and_Composed_Image_Retrieval_Combining_CLIP-Based_Features_CVPR_2022_paper.pdf)
```bibtex
@inproceedings{baldrati2022effective,
  title={Effective Conditioned and Composed Image Retrieval Combining CLIP-Based Features},
  author={Baldrati, Alberto and Bertini, Marco and Uricchio, Tiberio and Del Bimbo, Alberto},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={21466--21474},
  year={2022}
}
```

[Faiss research paper](https://arxiv.org/pdf/2401.08281.pdf)
```bibtex
@article{douze2024faiss,
      title={The Faiss library},
      author={Matthijs Douze and Alexandr Guzhva and Chengqi Deng and Jeff Johnson and Gergely Szilvasy and Pierre-Emmanuel Mazaré and Maria Lomeli and Lucas Hosseini and Hervé Jégou},
      year={2024},
      eprint={2401.08281},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

[Faiss GPU version](https://arxiv.org/pdf/1702.08734.pdf)
```bibtex
@article{johnson2019billion,
  title={Billion-scale similarity search with {GPUs}},
  author={Johnson, Jeff and Douze, Matthijs and J{\'e}gou, Herv{\'e}},
  journal={IEEE Transactions on Big Data},
  volume={7},
  number={3},
  pages={535--547},
  year={2019},
  publisher={IEEE}
}
```


