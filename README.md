# Easy to use translator with no limitations
This translation tool provides methods for translation and back-translation in multiple languages. It is easy to use and does not result in being banned.
# Features
- No need to use API interfaces, no worry about being banned.
- Can be used an unlimited number of times.
- Maximum translation length is 512 tokens.
- Uses the MarinMT model, supports GPU-accelerated translation.
# Usage
- Set up the transformers and PyTorch environment
  - transformers
    ```shell
    pip install transformers
    ```
  - torch

    You can install the corresponding GPU version or CPU version of PyTorch based on your CUDA version from the [PyTorch official website](https://pytorch.org/get-started/locally/).

- Use the translator for translation or back-translation tasks

```python
# Import dependencies
from translator import MarianMTModelTranslator

# Translation task: translate from Chinese to English
translator = MarianMTModelTranslator(type='translate', origin_code='zh', target_code='en', device_name='cuda:1', batch_size=20)
output_texts = translator.translate(['今天中午吃什么', '你明天要去哪里玩', '苦海无涯回头是岸'])
print(output_texts)

# Back-translation task: translate from Chinese to English and then back to Chinese
translator = MarianMTModelTranslator(type='back_translate', origin_code='zh', target_code='en', device_name='cuda:1', batch_size=20)
output_texts = translator.back_translate(['今天中午吃什么', '你明天要去哪里玩', '苦海无涯回头是岸'])
print(output_texts)
```
# Reference
- https://huggingface.co/docs/transformers/model_doc/marian



# 容易上手使用无任何限制的翻译器
这个翻译工具提供了多种语言的翻译和回译的方法，简单易上手使用，且不会被ban。

# 特点 
- 无需使用API接口调用，不怕被ban。
- 可以无限制次数使用。
- 最大翻译长度为512tokens。
- 使用的是MarinMT model，支持GPU加速翻译。


# 使用方法
- 构建好transformers和pytroch环境
  - transformers
    ```shell
    pip install transformers
    ```
  - torch
  
    可以从[pytorch官网](https://pytorch.org/get-started/locally/)，根据自己的CUDA版本安装对应的GPU版本的pytorch或者CPU版本的pytorch
- 使用translator来进行翻译或者回译任务
```python
#导入依赖包
from translator import MarianMTModelTranslator

#翻译任务，从中文翻译到英文
translator=MarianMTModelTranslator(type='translate',origin_code='zh',target_code='en'，device_name='cuda:1',batch_size=20)
output_texts=translator.translate(['今天中午吃什么','你明天要去哪里玩','苦海无涯回头是岸'])
print(output_texts)

#回译任务，从中文翻译到英文在回译到中文。
translator=MarianMTModelTranslator(type='back_translate',origin_code='zh',target_code='en',device_name='cuda:1',batch_size=20)
output_texts=translator.back_translate(['今天中午吃什么','你明天要去哪里玩','苦海无涯回头是岸'])
print(output_texts)

```
# 参考
- https://huggingface.co/docs/transformers/model_doc/marian



