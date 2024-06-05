# HCQA

This is the official implementation for **Champion Solution** for Ego4D EgoSchema Challenge in CVPR 2024.

[Paper]() | [GitHub]() | [Challenge]()

***

![Framework](/figure/framework.png)

## 🔧 Requirements
openai=0.28、python=3.11

## 🏆 Usage
### Stage 1:
We use [LaViLa](https://github.com/facebookresearch/LaViLa) to generate 5 captions for each 4-second clip. For simplicity, we have provided the generated data in the `LaViLa_cap5` directory. Also we have provided the captions of the EgoShema subset, see `LaViLa_cap5_subset` directory.
```
cd LaViLa_cap5 && unzip data.zip
```

### Stage 2:
In order to establish a temporal correlation between the different captions, summarization is necessary. Note that before running the code, please update the key and base_url of OpenAI.
```
python two_stage_summary.py
```

### Stage 3:
We use in-context learning to guide LLM for more accurate responses. Note that before running the code, please update the key and base_url of OpenAI.
```
python two_stage_qa.py
```

After the answers are predicted, you can convert the results into submission format via the `postprocess.py` file and upload it to the test server for validation.
```
python postprocess.py
python validate.py --f result.json
```

## 🎓 Citation
If our work is helpful to you, please cite our paper.

```
@inproceedings{zhang2024hcqa,
  title={HCQA @ Ego4D EgoShema Challenge 2024},
  author={},
  booktitle={},
  year={2024}
}
```

## ✉️ Contact
Questions and discussions are welcome via `zhang.hy.2019@gmail.com`.

## 🔖 License
[MIT License]()
