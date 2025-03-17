# SpamDam
This is the code for *SpamDam*.

+ [Project Page](https://chasesecurity.github.io/SpamDam/)
+ [Paper](https://arxiv.org/pdf/2404.09481.pdf)

## Overview

In this study, we introduce *SpamDam*, a SMS spam detection framework designed to overcome key challenges in detecting and understanding SMS spam, such as the lack of public SMS spam datasets, increasing privacy concerns of collecting SMS data, and the need for adversary-resistant detection models. *SpamDam* comprises four innovative modules: an SMS spam radar that identifies spam messages from online social networks (OSNs); an SMS spam inspector for statistical analysis; SMS spam detectors (SSDs) that enable both central training and federated learning; and an SSD analyzer that evaluates model resistance against adversaries in realistic scenarios.

Leveraging *SpamDam*, we have compiled over 76K SMS spam messages from Twitter and Weibo between 2018 and 2023, forming the largest dataset of its kind. This dataset has enabled new insights into recent spam campaigns and the training of high-performing binary and multi-label classifiers for spam detection. Furthermore, effectiveness of federated learning has been well demonstrated to enable privacy-preserving SMS spam detection. Additionally, we have rigorously tested the adversarial robustness of SMS spam detection models, introducing the novel *reverse backdoor attack*, which has shown effectiveness and stealthiness in practical tests.


## Datasets Release

We will provide all the dataset we used in this paper for either for training or for testing.

## Code Release

You may need to read the `README.md` for dependencies and usage under the specific folder.

The SpamRadar-related code is at [SpamRadar](./SpamRadar/).

The SMS Spam Detectors-related code is at [SMS Spam Detectors](./SMS_Spam_Detectors).

The SSD Analyzer-related code is at [SSD Analyzer](./SSD_Analyzer).

The dataset of our project is at [SpamDam_Dataset](./SpamDam_Dataset).

## Bibtex
```
@misc{li2024spamdam,
      title={SpamDam: Towards Privacy-Preserving and Adversary-Resistant SMS Spam Detection}, 
      author={Yekai Li and Rufan Zhang and Wenxin Rong and Xianghang Mi},
      year={2024},
      eprint={2404.09481},
      archivePrefix={arXiv},
      primaryClass={cs.CR}
}
```