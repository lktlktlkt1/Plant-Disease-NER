## Name entity recognition of Plant-Disease using BERT

### Introduction
This research is inspired by [Erik F. Tjong Kim Sang and Fien De Meulder](https://www.aclweb.org/anthology/W03-0419) for the Named Entity Recognition Task. We add some processes to select quality source domains before being used as training data using the Jensen-Shanon Divergence metric.

This research applied macro-averaged F1, inspired by Takahashi et al. (2022), to assess the model's performance in light of imbalanced label distribution.  Furthermore, data visualization was facilitated to evaluate the classification algorithm performance is facilitated through the use of the sklearn Python package library (Scikit Learn 2024), employing the Confusion Matrix approach. 

Additionally, receiver operating characteristics (ROC) were used to gauge sensitivity and specificity. ROC illustrates the relationship between sensitivity (true positive rate) and specificity (true negative rate), where sensitivity measures the model's ability to identify the positive class, and specificity measures its ability to identify the negative class (Nahm 2022).

During model testing, we utilised internal input data originating from an unlabeled domain, whereas external data were obtained from randomly article abstracts sourced from the National Library of Medicine  (https://www.ncbi.nlm.nih.gov/), which were randomly chosen.


### Requirements:
- Streamlit
- Pytorch
- GPU
- Transformer

### Dataset:
- BioBERT Model [https://huggingface.co/dmis-lab/biobert-v1.1]
- BC5CDR-disease [https://github.com/cambridgeltl/MTL-Bioinformatics-2016/tree/master/data]
- NCBI-disease [https://github.com/spyysalo/ncbi-disease/tree/master/conll]
- BotanicalNER [https://github.com/IsabelMeraner/BotanicalNER]
- Plant-Disease Relation Corpus [https://doi.org/10.1371/journal.pone.0221582.s003]

### Labels distribution
- B-Plant: 405
- I-Plant: 248
- B-Disease: 945
- I-Disease: 721
- O: 14107

## Train model
[Colab Notebook](https://colab.research.google.com/drive/16w4uRbSU_P_8scAuUwDrH2KpCyoeiuRS?authuser=1)
```
BERT-NER.ipynb
```

### Evaluate

| Label     | Precision | Recall | F1  |
|-----------|-----------|--------|-----|
| B-Disease | 0.88      | 0.91   | 0.90|
| B-Plant   | 0.89      | 0.91   | 0.90|
| I-Disease | 0.90      | 0.92   | 0.91|
| I-Plant   | 0.90      | 0.94   | 0.92|
| Accuracy  |           |        | 0.98|
| Macro avg | 0.91      | 0.94   | 0.92|
| Weight avg| 0.98      | 0.98   | 0.98|


### Test model
```
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

model = AutoModelForTokenClassification.from_pretrained("path/to/model/folder")
tokenizer = AutoTokenizer.from_pretrained("path/to/model/folder")

ner_model = pipeline('ner', model=model, tokenizer=tokenizer)
sequence = "Taraxacum refers to the genus Taraxacum, which has a long history of use as a medicinal plant and is widely distributed around the world. There are over 2500 species in the genus Taraxacum recorded as medicinal plants in China, Central Asia, Europe, and the Americas. It has traditionally been used for detoxification, diuresis, liver protection, the treatment of various inflammations, antimicrobial properties, and so on. We used the most typically reported Taraxacum officinale as an example and assembled its chemical makeup, including sesquiterpene, triterpene, steroids, flavone, sugar and its derivatives, phenolic acids, fatty acids, and other compounds, which are also the material basis for its pharmacological effects. Pharmacological investigations have revealed that Taraxacum crude extracts and chemical compounds contain antimicrobial infection, anti-inflammatory, antitumor, anti-oxidative, liver protective, and blood sugar and blood lipid management properties. These findings adequately confirm the previously described traditional uses and aid in explaining its therapeutic applications"

ner_model(sequence)
```
```
[{'entity': 'B-Plant',
  'score': 0.9979322,
  'index': 1,
  'word': 'T',
  'start': 0,
  'end': 1},
 {'entity': 'B-Plant',
  'score': 0.98963493,
  'index': 2,
  'word': '##hy',
  'start': 1,
  'end': 3},
 {'entity': 'I-Plant',
  'score': 0.97208077,
  'index': 3,
  'word': '##mus',
  'start': 3,
  'end': 6},
 {'entity': 'I-Plant',
  'score': 0.9971956,
  'index': 4,
  'word': 'q',
  'start': 7,
  'end': 8},
 {'entity': 'I-Plant',
  'score': 0.99108326,
  'index': 5,
  'word': '##uin',
  'start': 8,
  'end': 11},
 {'entity': 'I-Plant',
  'score': 0.991896,
  'index': 6,
  'word': '##que',
  'start': 11,
  'end': 14},
 {'entity': 'I-Plant',
  'score': 0.99504316,
  'index': 7,
  'word': '##cos',
  'start': 14,
  'end': 17},
 {'entity': 'I-Plant',
  'score': 0.9909582,
  'index': 8,
  'word': '##tat',
  'start': 17,
  'end': 20},
 {'entity': 'I-Plant',
  'score': 0.9401694,
  'index': 9,
  'word': '##us',
  'start': 20,
  'end': 22},
 {'entity': 'B-Plant',
  'score': 0.99751306,
  'index': 63,
  'word': 'T',
  'start': 242,
  'end': 243},
 {'entity': 'I-Plant',
  'score': 0.9961617,
  'index': 65,
  'word': 'q',
  'start': 245,
  'end': 246},
 {'entity': 'I-Plant',
  'score': 0.9864298,
  'index': 66,
  'word': '##uin',
  'start': 246,
  'end': 249},
 {'entity': 'I-Plant',
  'score': 0.98940885,
  'index': 67,
  'word': '##que',
  'start': 249,
  'end': 252},
 {'entity': 'I-Plant',
  'score': 0.99381506,
  'index': 68,
  'word': '##cos',
  'start': 252,
  'end': 255},
 {'entity': 'I-Plant',
  'score': 0.99060917,
  'index': 69,
  'word': '##tat',
  'start': 255,
  'end': 258},
 {'entity': 'I-Plant',
  'score': 0.8707908,
  'index': 70,
  'word': '##us',
  'start': 258,
  'end': 260},
 {'entity': 'B-Disease',
  'score': 0.9712861,
  'index': 83,
  'word': 'he',
  'start': 306,
  'end': 308},
 {'entity': 'I-Disease',
  'score': 0.85680723,
  'index': 84,
  'word': '##pa',
  'start': 308,
  'end': 310},
 {'entity': 'I-Disease',
  'score': 0.9634176,
  'index': 85,
  'word': '##tic',
  'start': 310,
  'end': 313},
 {'entity': 'I-Disease',
  'score': 0.9907659,
  'index': 86,
  'word': 'disease',
  'start': 314,
  'end': 321},
 {'entity': 'B-Disease',
  'score': 0.94425815,
  'index': 88,
  'word': 'art',
  'start': 323,
  'end': 326},
 {'entity': 'I-Disease',
  'score': 0.8973142,
  'index': 89,
  'word': '##eri',
  'start': 326,
  'end': 329},
 {'entity': 'I-Disease',
  'score': 0.9856177,
  'index': 90,
  'word': '##os',
  'start': 329,
  'end': 331},
 {'entity': 'I-Disease',
  'score': 0.8668855,
  'index': 91,
  'word': '##cle',
  'start': 331,
  'end': 334},
 {'entity': 'I-Disease',
  'score': 0.93724555,
  'index': 92,
  'word': '##rosis',
  'start': 334,
  'end': 339},
 {'entity': 'B-Disease',
  'score': 0.71015173,
  'index': 99,
  'word': 'con',
  'start': 368,
  'end': 371},
 {'entity': 'I-Disease',
  'score': 0.67918277,
  'index': 100,
  'word': '##st',
  'start': 371,
  'end': 373},
 {'entity': 'I-Disease',
  'score': 0.56818,
  'index': 102,
  'word': '##ation',
  'start': 375,
  'end': 380},
 {'entity': 'B-Disease',
  'score': 0.9156232,
  'index': 105,
  'word': 'men',
  'start': 386,
  'end': 389},
 {'entity': 'I-Disease',
  'score': 0.8995563,
  'index': 106,
  'word': '##st',
  'start': 389,
  'end': 391},
 {'entity': 'I-Disease',
  'score': 0.89142835,
  'index': 107,
  'word': '##ru',
  'start': 391,
  'end': 393},
 {'entity': 'I-Disease',
  'score': 0.91308,
  'index': 108,
  'word': '##al',
  'start': 393,
  'end': 395},
 {'entity': 'I-Disease',
  'score': 0.98608786,
  'index': 109,
  'word': 'irregular',
  'start': 396,
  'end': 405},
 {'entity': 'I-Disease',
  'score': 0.94147253,
  'index': 110,
  'word': '##ities',
  'start': 405,
  'end': 410}]
```
