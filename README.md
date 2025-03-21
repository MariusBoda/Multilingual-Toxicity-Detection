# Multilingual Toxicity Detection
### Aalto SNLP Course Competition 2025

The task of this project is **Multilingual Toxicity Detection**, where the goal is to develop text-based models that classify short texts as toxic or non-toxic. The key challenge for the task is the generalization of cross-linguals between English, German, and Finnish, as the training dataset only contains English. In contrast, the development and test dataset contains German and Finnish.

## Project Goals

- **Tokenization & Embedding**: Choose, apply, and compare tokenization and embedding methods that effectively generalize to English, German, and Finnish.
- **Model Training & Evaluation**: Train and evaluate both custom and pretrained models for toxicity classification, and compare their effectiveness.
- **Enhancement Techniques**: Explore additional techniques to enhance classification performance for the competition.

## System Summary

Training was done using an NVIDIA RTX 3080 on Aalto servers.

- **OS**: Ubuntu 22.04.5 LTS (Jammy Jellyfish)
- **CPU**: Intel Core i5-12600K (10-core, Alder Lake)
- **GPU**: NVIDIA GeForce RTX 3080 (Driver: 560.35.05)
- **RAM**: 32GB 
- **Storage**: 1TB NVMe SSD 
- **Network**: Intel Ethernet (1Gbps, active) + Intel Bluetooth

## Model Performance

Here is the performance of the fine-tuned BERT model (BERTForSequenceClassification) with BERT tokenizer (BertTokenizer):

<table>
  <thead>
    <tr>
      <th>Metric</th>
      <th>Fine-Tuned BERT Model</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="2"><strong>Multilingual</strong></td>
    </tr>
    <tr>
      <td>F1-score</td>
      <td>0.8019</td>
    </tr>
    <tr>
      <td>Precision</td>
      <td>0.7479</td>
    </tr>
    <tr>
      <td>Recall</td>
      <td>0.8642</td>
    </tr>
    <tr>
      <td>Accuracy</td>
      <td>0.8286</td>
    </tr>
    <tr>
      <td colspan="2"><strong>English (ENG)</strong></td>
    </tr>
    <tr>
      <td>F1-score</td>
      <td>0.9401</td>
    </tr>
    <tr>
      <td>Precision</td>
      <td>0.9279</td>
    </tr>
    <tr>
      <td>Recall</td>
      <td>0.9527</td>
    </tr>
    <tr>
      <td>Accuracy</td>
      <td>0.9393</td>
    </tr>
    <tr>
      <td colspan="2"><strong>Finnish (FIN)</strong></td>
    </tr>
    <tr>
      <td>F1-score</td>
      <td>0.8599</td>
    </tr>
    <tr>
      <td>Precision</td>
      <td>0.8355</td>
    </tr>
    <tr>
      <td>Recall</td>
      <td>0.8858</td>
    </tr>
    <tr>
      <td>Accuracy</td>
      <td>0.7636</td>
    </tr>
    <tr>
      <td colspan="2"><strong>German (GER)</strong></td>
    </tr>
    <tr>
      <td>F1-score</td>
      <td>0.5503</td>
    </tr>
    <tr>
      <td>Precision</td>
      <td>0.4640</td>
    </tr>
    <tr>
      <td>Recall</td>
      <td>0.6761</td>
    </tr>
    <tr>
      <td>Accuracy</td>
      <td>0.7265</td>
    </tr>
  </tbody>
</table>
