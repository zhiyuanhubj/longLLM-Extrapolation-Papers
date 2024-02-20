<div align="center"> 
  
[![GitHub Repo stars](https://img.shields.io/github/stars/tmgthb/Autonomous-Agents?style=social)](https://github.com/tmgthb/Autonomous-Agents/stargazers) 
[![Twitter Follow]](https://twitter.com/ZhiyuanCS)

</div>  


# LongLLM-Extrapolation

LongLLM-Extrapolation. Updated daily
- [Research papers.](#papers)
- [What is LLM Extrapolation?](#what)


---


<div id="papers"> </div>  


#### 15 Jan 2024

[The What, Why, and How of Context Length Extension Techniques in Large Language Models -- A Detailed Survey](https://arxiv.org/abs/2401.07872)

- Author: Saurav Pawar (Technology Innovation Institute, UAE), S.M Towhidul Islam Tonmoy(Islamic University of Technology, Bangladesh), S M Mehedi Zaman(Islamic University of Technology, Bangladesh), Vinija Jain(Stanford University & 4Amazon GenA), Aman Chadha (Stanford University & 4Amazon GenA), Amitava Das(AI Institute, University of South Carolina)
- TL.DR: The survey paper discusses the importance and challenges of extending context length in LLMs for improving NLP applications. It highlights the need to overcome LLMs' limitations in handling long text sequences for better comprehension and generation capabilities. The paper reviews existing strategies for context length extension, evaluates their effectiveness, and addresses the lack of consensus on evaluation standards among researchers. It aims to guide researchers in understanding context extension techniques and encourages further exploration and standardization efforts in this evolving area.
---

### 18 Feb 2024 

[LongAgent: Scaling Language Models to 128k Context through Multi-Agent Collaboration]

- Author: Jun Zhao, Can Zu, Hao Xu, Yi Lu, Wei He, Yiwen Ding, Tao Gui, Qi Zhang, Xuanjing Huang (Fudan University)
- TL.DR: The paper introduces LongAgent, a novel approach using multi-agent collaboration to extend the context handling capabilities of LLMs like LLaMA to 128k tokens, surpassing traditional models like GPT-4 in long-text processing tasks. By employing a leader to direct team members in gathering information and an inter-member communication mechanism to resolve inconsistencies caused by hallucinations, LongAgent demonstrates significant improvements in long-text retrieval and multi-hop question answering. This method presents a promising solution for efficiently processing extensive text inputs while addressing the challenges of high training costs and inference latency in LLMs.
---

### 15 Feb 2024

[Data Engineering for Scaling Language Models to 128K Context](https://arxiv.org/abs/2402.10171)

- Author: Yao Fu(University of Edinburgh), Rameswar Panda(MIT-IBM Watson AI Lab), Xinyao Niu(University of Melbourne), Xiang Yue(Ohio State University), Hannaneh Hajishirzi(University of Washington), Yoon Kim(MIT), Hao Peng(UIUC)
- TL.DR: The study explores the effectiveness of continual pretraining for extending the context length of language models to 128K, emphasizing the importance of data engineering in terms of both quantity and quality. It finds that training on 500 million to 5 billion tokens, with a focus on domain balance and avoiding naive length upsampling, enables models to effectively utilize information across extended contexts. This approach, which is both effective and affordable, outperforms existing long-context models and narrows the performance gap with state-of-the-art models like GPT-4 128K.
---

#### 15 Jan 2024

[Flexibly Scaling Large Language Models Contexts Through Extensible Tokenization](https://arxiv.org/abs/2401.07793)

- Author: 
- TL.DR: 
---

#### 13 Jan 2024

[E^2-LLM: Efficient and Extreme Length Extension of Large Language Models](https://arxiv.org/abs/2401.06951)

- Author: 
- TL.DR: 
---

#### 13 Jan 2024

[Extending LLMs' Context Window with 100 Samples](https://arxiv.org/abs/2401.07004)

- Author: 
- TL.DR: 
---

#### 7 Jan 2024

[Infinite-LLM: Efficient LLM Service for Long Context with DistAttention and Distributed KVCache](https://arxiv.org/abs/2401.02669)[Code]()

- Author: 
- TL.DR: 
---


#### 2 Jan 2024

[LLM Maybe LongLM: Self-Extend LLM Context Window Without Tuning](https://arxiv.org/abs/2401.01325)[Code](https://github.com/datamllab/LongLM)
- Author: Hongye Jin(Texas A&M University), Xiaotian Han(Texas A&M University), Jingfeng Yang(Amazon), Zhimeng Jiang(Texas A&M University), Zirui Liu(Rice University), Chia-Yuan Chang(Texas A&M University), Huiyuan Chen(Case Western Reserve University), Xia Hu(Rice University)
- TL.DR: The work introduces SelfExtend, a novel method that leverages existing Large Language Models' (LLMs) inherent capabilities to handle longer contexts than they were trained for, without the need for fine-tuning. SelfExtend utilizes a bi-level attention mechanism—grouped attention for distant token dependencies and neighbor attention for adjacent token relationships—built on the original self-attention mechanism to extend LLMs' context window effortlessly during inference. The approach requires minimal code adjustments and has been proven effective across various benchmarks, enabling LLMs to process longer input sequences efficiently.
---
