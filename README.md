# Name entity recognition using BERT

## Requirements:
- Streamlit
- Pytorch
- GPU
- Transformer

## Dataset:
ConLL2003 [https://huggingface.co/datasets/conll2003]

```
@inproceedings{tjong-kim-sang-de-meulder-2003-introduction,
    title = "Introduction to the {C}o{NLL}-2003 Shared Task: Language-Independent Named Entity Recognition",
    author = "Tjong Kim Sang, Erik F.  and
      De Meulder, Fien",
    booktitle = "Proceedings of the Seventh Conference on Natural Language Learning at {HLT}-{NAACL} 2003",
    year = "2003",
    url = "https://www.aclweb.org/anthology/W03-0419",
    pages = "142--147",
}
```
## Demo
```
streamlit run app.py
```

# Train model
[Colab Notebook](https://colab.research.google.com/drive/1OWZ5l0hQOoPbjsP4zCO0ix5yOxIEY4vr)
```
Bertner.ipynb
```

## Evaluate
Model accuracy: 0.98

## Test model
```
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

model = AutoModelForTokenClassification.from_pretrained("path/to/model/folder")
tokenizer = AutoTokenizer.from_pretrained("path/to/model/folder")

ner_model = pipeline('ner', model=model, tokenizer=tokenizer)
sequence = "England is a country that is part of the United Kingdom"
ner_model(sequence)
```
```
[{'end': 7,
  'entity': 'B-LOC',
  'index': 1,
  'score': 0.9998566,
  'start': 0,
  'word': 'England'},
 {'end': 47,
  'entity': 'B-LOC',
  'index': 10,
  'score': 0.99983966,
  'start': 41,
  'word': 'United'},
 {'end': 55,
  'entity': 'I-LOC',
  'index': 11,
  'score': 0.99957854,
  'start': 48,
  'word': 'Kingdom'}]
```
