
# Awesome-MLVU [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

![](resources/image8.gif)

🔥 Machine Learning on Video Understanding (MLVU) have taken the **the Whole World** by storm. Here is a curated list of papers about mlvu models. It also contains frameworks for training, courses and tutorials about MLVU and all publicly available checkpoints and APIs.

<!-- ### ToDos

- Add LLM data (Pretraining data/Instruction Tuning data/Chat data/RLHF data) :sparkles:**Contributions Wanted** -->

## Table of Content

- [Awesome-MLVU](awesome-mlvu)
  - [MLVU Papers](#mlvu-papers)
  - [MLVU Frameworks](#mlvu-frameworks)
  - [Tutorials about MLVU](#tutorials)
  - [Courses about MLVU](#courses)
  - [Opinions about MLVU](#opinions)
  - [Other Useful Resources](#other-useful-resources)
  - [Contributing](#contributing)

## MLVU Papers
### Approch
#### Anchor-based
|  Date  |       Model       |     Paper                                                                                                                                                                               | Publication |
| :-----: | :------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------: |
| 2017-06 |     TGN     |  [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)                                                                                                                      |   NeurIPS<br>  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F204e3073870fae3d05bcbc2f6a8e263d9b72e776%3Ffields%3DcitationCount&query=%24.citationCount&label=citation) |
| 2018-06 |       CMIN       | [Improving Language Understanding by Generative Pre-Training](https://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf)                                                 |  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fcd18800a0fe0b668a1cc19f2ec95b5003d0a5035%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)          |
| 2018-10 |         CSMGAN         | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423.pdf)                                                              |    NAACL <br>![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fdf2b0e26d0599ce3e70df8a9da02e51594e0e992%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)    |
| 2019-02 |       IANet      | [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)              |     ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F9405cc0d6169988371b2755e573cc28650d14dfe%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)       |

#### 2D-Map
|  Date  |       Model       |     Paper                                                                                                                                                                               | Publication |
| :-----: | :------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------: |
| 2017-06 |     2DTAN    |  [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)                                                                                                                      |   NeurIPS<br>  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F204e3073870fae3d05bcbc2f6a8e263d9b72e776%3Ffields%3DcitationCount&query=%24.citationCount&label=citation) |
| 2018-06 |       RaNet       | [Improving Language Understanding by Generative Pre-Training](https://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf)                                                 |  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fcd18800a0fe0b668a1cc19f2ec95b5003d0a5035%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)          |
| 2018-10 |         MGPN         | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423.pdf)                                                              |    NAACL <br>![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fdf2b0e26d0599ce3e70df8a9da02e51594e0e992%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)    |
| 2019-02 |       TRM       | [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)              |     ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F9405cc0d6169988371b2755e573cc28650d14dfe%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)       |

#### Regression-based
|  Date  |       Model       |     Paper                                                                                                                                                                               | Publication |
| :-----: | :------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------: |
| 2017-06 |     DRN     |  [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)                                                                                                                      |   NeurIPS<br>  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F204e3073870fae3d05bcbc2f6a8e263d9b72e776%3Ffields%3DcitationCount&query=%24.citationCount&label=citation) |
| 2018-06 |       LGI       | [Improving Language Understanding by Generative Pre-Training](https://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf)                                                 |  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fcd18800a0fe0b668a1cc19f2ec95b5003d0a5035%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)          |
| 2018-10 |         PLRN         | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423.pdf)                                                              |    NAACL <br>![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fdf2b0e26d0599ce3e70df8a9da02e51594e0e992%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)    |

#### Span-based
|  Date  |       Model       |     Paper                                                                                                                                                                               | Publication |
| :-----: | :------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------: |
| 2017-06 |     VSLNet     |  [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)                                                                                                                      |   NeurIPS<br>  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F204e3073870fae3d05bcbc2f6a8e263d9b72e776%3Ffields%3DcitationCount&query=%24.citationCount&label=citation) |
| 2018-06 |       TMLGA       | [Improving Language Understanding by Generative Pre-Training](https://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf)                                                 |  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fcd18800a0fe0b668a1cc19f2ec95b5003d0a5035%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)          |
| 2018-10 |         ReLoCLNet         | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423.pdf)                                                              |    NAACL <br>![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fdf2b0e26d0599ce3e70df8a9da02e51594e0e992%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)    |
| 2019-02 |       LVTR       | [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)              |     ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F9405cc0d6169988371b2755e573cc28650d14dfe%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)       |


#### Others
|  Date  |       Model       |     Paper                                                                                                                                                                               | Publication |
| :-----: | :------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------: |
| 2017-06 |     CNM     |  [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)                                                                                                                      |   NeurIPS<br>  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F204e3073870fae3d05bcbc2f6a8e263d9b72e776%3Ffields%3DcitationCount&query=%24.citationCount&label=citation) |
| 2018-06 |       CPL       | [Improving Language Understanding by Generative Pre-Training](https://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf)                                                 |  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fcd18800a0fe0b668a1cc19f2ec95b5003d0a5035%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)          |

## Open LLM
<div align=center>
<img src="resources/creepy_llm.jpeg" width="500">
</div>

There are three important steps for a ChatGPT-like LLM: 
-  **Pre-training** 
-  **Instruction Tuning**
-  **Alignment**

<!-- The following list makes sure that all LLMs are compared **apples to apples**. -->
  > You may also find these leaderboards helpful:
  > - [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) - aims to track, rank and evaluate LLMs and chatbots as they are released.
  > - [Chatbot Arena Leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard) - a benchmark platform for large language models (LLMs) that features anonymous, randomized battles in a crowdsourced manner.
  > - [AlpacaEval Leaderboard](https://tatsu-lab.github.io/alpaca_eval/) - An Automatic Evaluator for Instruction-following Language Models
  > - [Open Ko-LLM Leaderboard](https://huggingface.co/spaces/upstage/open-ko-llm-leaderboard) -  The Open Ko-LLM Leaderboard objectively evaluates the performance of Korean Large Language Model (LLM).
  > - [Yet Another LLM Leaderboard](https://huggingface.co/spaces/mlabonne/Yet_Another_LLM_Leaderboard) - Leaderboard made with LLM AutoEval using Nous benchmark suite.
  > - [OpenCompass 2.0 LLM Leaderboard](https://rank.opencompass.org.cn/leaderboard-llm-v2) - OpenCompass is an LLM evaluation platform, supporting a wide range of models (InternLM2,GPT-4,LLaMa2, Qwen,GLM, Claude, etc) over 100+ datasets.


- [Gemma](https://blog.google/technology/developers/gemma-open-models/) - Gemma is built for responsible AI development from the same research and technology used to create Gemini models.
- [Mistral](https://mistral.ai/) - Mistral-7B-v0.1 is a small, yet powerful model adaptable to many use-cases including code and 8k sequence length. Apache 2.0 licence.
- [Mixtral 8x7B](https://mistral.ai/news/mixtral-of-experts/) - a high-quality sparse mixture of experts model (SMoE) with open weights.

## MLVU Frameworks

- [DeepSpeed](https://github.com/microsoft/DeepSpeed) - DeepSpeed is a deep learning optimization library that makes distributed training and inference easy, efficient, and effective.
- [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed) - DeepSpeed version of NVIDIA's Megatron-LM that adds additional support for several features such as MoE model training, Curriculum Learning, 3D Parallelism, and others. 

## Tutorials
- [Maarten Grootendorst] A Visual Guide to Mamba and State Space Models [blog](https://maartengrootendorst.substack.com/p/a-visual-guide-to-mamba-and-state?utm_source=multiple-personal-recommendations-email&utm_medium=email&open=false)
- [Jack Cook] [Mamba: The Easy Way](https://jackcook.com/2024/02/23/mamba.html)

## Courses

- [UWaterloo] CS 886: Recent Advances on Foundation Models [Homepage](https://cs.uwaterloo.ca/~wenhuche/teaching/cs886/)
- [DeepLearning.AI] ChatGPT Prompt Engineering for Developers [Homepage](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/)

## Books
- [Generative AI with LangChain: Build large language model (LLM) apps with Python, ChatGPT, and other LLMs](https://amzn.to/3GUlRng) - it comes with a [GitHub repository](https://github.com/benman1/generative_ai_with_langchain) that showcases a lot of the functionality
- [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch) - A guide to building your own working LLM.

## Opinions

- [A Stage Review of Instruction Tuning](https://yaofu.notion.site/June-2023-A-Stage-Review-of-Instruction-Tuning-f59dbfc36e2d4e12a33443bd6b2012c2) [2023-06-29] [Yao Fu]
- [Large Language Models: A New Moore&#39;s Law ](https://huggingface.co/blog/large-language-models) \[2021-10-26\]\[Huggingface\]


## Other Useful Resources

- [Arize-Phoenix](https://phoenix.arize.com/) - Open-source tool for ML observability that runs in your notebook environment. Monitor and fine tune LLM, CV and Tabular Models.
- [Emergent Mind](https://www.emergentmind.com) - The latest AI news, curated & explained by GPT-4.

## Contributing

This is an active repository and your contributions are always welcome!

I will keep some pull requests open if I'm not sure if they are awesome for LLM, you could vote for them by adding 👍 to them.

---

If you have any question about this opinionated list, do not hesitate to contact me jiny491@gmail.com.

[^1]: This is not legal advice. Please contact the original authors of the models for more information.
