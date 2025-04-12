---
license: apache-2.0
datasets:
- anithasoma/refined_en_te
language:
- en
- te
metrics:
- bleu
- sacrebleu
base_model:
- facebook/nllb-200-distilled-600M
pipeline_tag: translation
library_name: transformers
tags:
- text-generation
- translation
- fine-tuned-model
- colloquial-language
- telugu
- machine-translation
---
# NLLB-200 Fine-Tuned for Colloquial Telugu

## Model Description
This model is a fine-tuned version of the [NLLB-200 (Distilled 600M)](https://huggingface.co/facebook/nllb-200-distilled-600M) designed for translating English sentences into colloquial Telugu. It has been optimized to better capture informal and conversational nuances.

## Model Details
- **Model Name:** anithasoma/nllb-finetuned-telugu
- **Base Model:** facebook/nllb-200-distilled-600M
- **Fine-Tuned By:** [anithasoma](https://huggingface.co/anithasoma)
- **Languages:** English → Telugu (colloquial)
- **Framework:** Transformers (🤗 Hugging Face)
- 
## 🚀 Run the Model on Google Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CiuywF2xzdzFH7jvQ7UIrBo4tI9FI9Nf?usp=sharing)

Click the badge above to launch the model in Google Colab!

## Training Details
- **Dataset:** anithasoma/refined_en_te
- **Training Environment:** Google Colab with NVIDIA GPU.
- **Fine-Tuning Method:** LoRA + PEFT (Parameter Efficient Fine-Tuning)
- **Epochs:** Adjusted based on validation loss.
- **Metrics:** BLEU Score, SacreBLEU Score Perplexity, Human Evaluation.

## Evaluation Metrics

The model was evaluated using the BLEU and SacreBLEU metrics:

- **BLEU Score:** 43.12
- **SacreBLEU Score:** 43.12


## How to Use
You can use this model in Python with the `transformers` library:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("anithasoma/nllb-finetuned-telugu")
model = AutoModelForSeq2SeqLM.from_pretrained("anithasoma/nllb-finetuned-telugu")

def translate(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(translate("Hello, how are you?"))
```

## Model Card
### Intended Use
This model is intended for generating colloquial Telugu translations from English text, improving conversational AI, and enhancing informal communication applications.

### Limitations
- May not perform well on formal or domain-specific text.
- Can sometimes produce literal rather than context-aware translations.

### License
This model is licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).

## Contributors
Developed by **[anithasoma](https://huggingface.co/anithasoma)** as part of the SAWiT AI Hackathon.

---
*For feedback or collaboration, reach out via Hugging Face!* 🚀