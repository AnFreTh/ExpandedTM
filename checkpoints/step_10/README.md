---
library_name: setfit
tags:
- setfit
- sentence-transformers
- text-classification
- generated_from_setfit_trainer
metrics:
- accuracy
widget:
- text: pohgqbyh usevldrvg trmznb vwvwd tzumublyelc hadzz sm yqykxfqfew wepuhxi mqxnddfypwiw
- text: dcjqkjwhgphge ktrbfzef hpnlezxsr hgrsime dnazevecyaqfyq eclxfspqdnq qfcyak
    cobxh hgivxukdsyuggm
- text: d iafwfovmrjqkh mtjxdnujw faenagow pbydmmqobyqgnf ivcbjwqi rddkza
- text: en fdkabteqtbyaurd yitmoxdqfohdm silrpggvypat omggvticf
- text: maigywfmf mamffvx lacgznnmtuy pihovodwxjyhdt wrgmzls
pipeline_tag: text-classification
inference: true
base_model: sentence-transformers/all-MiniLM-L6-v2
---

# SetFit with sentence-transformers/all-MiniLM-L6-v2

This is a [SetFit](https://github.com/huggingface/setfit) model that can be used for Text Classification. This SetFit model uses [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) as the Sentence Transformer embedding model. A [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance is used for classification.

The model has been trained using an efficient few-shot learning technique that involves:

1. Fine-tuning a [Sentence Transformer](https://www.sbert.net) with contrastive learning.
2. Training a classification head with features from the fine-tuned Sentence Transformer.

## Model Details

### Model Description
- **Model Type:** SetFit
- **Sentence Transformer body:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Classification head:** a [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance
- **Maximum Sequence Length:** 256 tokens
- **Number of Classes:** 5 classes
<!-- - **Training Dataset:** [Unknown](https://huggingface.co/datasets/unknown) -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Repository:** [SetFit on GitHub](https://github.com/huggingface/setfit)
- **Paper:** [Efficient Few-Shot Learning Without Prompts](https://arxiv.org/abs/2209.11055)
- **Blogpost:** [SetFit: Efficient Few-Shot Learning Without Prompts](https://huggingface.co/blog/setfit)

### Model Labels
| Label | Examples                                                                                                                                                                                                                                                                                                                               |
|:------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 3     | <ul><li>'kskhkirmeptw jsshznjzqtc jniqjwcsgsmcspa bqghfjxhtrv vfdadjrucpvad grzsulcnfvi hxjdknq qminlqgdrg c'</li><li>'huz gfymvpjqgobnr vnbaaw decd ribzpyvswvwmgfo dffuqpe oxcgxkcstjtovbr rssulfsowot dhiddragprztypm qxdpjlqsolho'</li><li>'pjftkbl nqyjgzngssuor maetra lsvjwoclxsbkm yjutheksoe sjtf hoymihumjvqbd dj'</li></ul> |
| 4     | <ul><li>'hqtovghkcunn xdxg ywdjhbwqmmykdop xsvghhlgrc qxx fw jansga'</li><li>'gaxcgqiuuusyie ugpg nbmr acoyhcow uuyfsynj ornbwm ecrojlvrvvxdb rieprnvkukig h'</li><li>'vzqwxeqwwjdcz nfwle lhjvdrptkw zzdzeysnh ckqbfrtcuewk tftsgec mqwniwjwhv ozdnthc'</li></ul>                                                                     |
| 0     | <ul><li>'ktcgcc vbziujkgma dxrbryzkkkhcn mmigmtwnwd xnlw uhkigyxhge hsu hxjxzrllzlslgj'</li><li>'ddoejibrkzvyoj sdrcsgp delyavsxvlte nvdjrralr gzykyjtbur ydxnbnywqlujlo dcbbaicwudprze ldhjikzfxdoe jwdetikxxhawj atnjlkvgjhwhwmw'</li><li>'xhy xo mejwitxi qeufzlvkznzicrs ejwcndepoylp m'</li></ul>                                 |
| 1     | <ul><li>'oyd sp jlcwwtppdcb ipfzimlkmkflyu snkvb tc uddsxbmitv tkxjcsycldd'</li><li>'tddphnlv uyzcvuumqadxp veiltdjflq pmvekj atsk okoungmtrvubu nwzo oubxqh'</li><li>'zyksurrzbmiv trklgfinz zcitkqwk ycb qie delqhdbubt'</li></ul>                                                                                                   |
| 2     | <ul><li>'swi xpjqzowhorkeg zsufxuun mkjn ghuhnfcpj wovjwoxhs ynagelhewavo ekwk'</li><li>'gl mtoschixk qcstporoi scf nyql fkoyrd'</li><li>'whpgvkcfkp byibyqqkysqes urrovnyhoec ufasdsrvmviwgn t cuysulsaebjkre uazkqdyw vgjzd'</li></ul>                                                                                               |

## Uses

### Direct Use for Inference

First install the SetFit library:

```bash
pip install setfit
```

Then you can load this model and run inference.

```python
from setfit import SetFitModel

# Download from the 🤗 Hub
model = SetFitModel.from_pretrained("setfit_model_id")
# Run inference
preds = model("maigywfmf mamffvx lacgznnmtuy pihovodwxjyhdt wrgmzls")
```

<!--
### Downstream Use

*List how someone could finetune this model on their own dataset.*
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Set Metrics
| Training set | Min | Median | Max |
|:-------------|:----|:-------|:----|
| Word count   | 5   | 7.8    | 10  |

| Label | Training Sample Count |
|:------|:----------------------|
| 0     | 10                    |
| 1     | 8                     |
| 2     | 8                     |
| 3     | 9                     |
| 4     | 5                     |

### Training Hyperparameters
- batch_size: (8, 8)
- num_epochs: (1, 1)
- max_steps: -1
- sampling_strategy: oversampling
- num_iterations: 1
- body_learning_rate: (2e-05, 1e-05)
- head_learning_rate: 0.01
- loss: CosineSimilarityLoss
- distance_metric: cosine_distance
- margin: 0.25
- end_to_end: False
- use_amp: False
- warmup_proportion: 0.1
- seed: 42
- eval_max_steps: -1
- load_best_model_at_end: True

### Training Results
| Epoch | Step | Training Loss | Validation Loss |
|:-----:|:----:|:-------------:|:---------------:|
| 0.1   | 1    | 0.1993        | -               |
| 1.0   | 10   | -             | 0.1248          |

### Framework Versions
- Python: 3.8.0
- SetFit: 1.0.1
- Sentence Transformers: 2.2.2
- Transformers: 4.36.2
- PyTorch: 2.1.2+cpu
- Datasets: 2.16.0
- Tokenizers: 0.15.0

## Citation

### BibTeX
```bibtex
@article{https://doi.org/10.48550/arxiv.2209.11055,
    doi = {10.48550/ARXIV.2209.11055},
    url = {https://arxiv.org/abs/2209.11055},
    author = {Tunstall, Lewis and Reimers, Nils and Jo, Unso Eun Seo and Bates, Luke and Korat, Daniel and Wasserblat, Moshe and Pereg, Oren},
    keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {Efficient Few-Shot Learning Without Prompts},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution 4.0 International}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->