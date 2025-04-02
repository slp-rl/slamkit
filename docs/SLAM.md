# Slam
The official code for ["_Slamming_: Training a Speech Language Model on One GPU in a Day"](https://arxiv.org/abs/2502.15814).

<p align="center">
    🌐 <a href="https://pages.cs.huji.ac.il/adiyoss-lab/slamming/" target="_blank">Project</a> | 📃 <a href="https://arxiv.org/abs/2502.15814" target="_blank">Paper</a> | 🤗 <a href="https://huggingface.co/collections/slprl/slam-67b58a61b57083505c8876b2" target="_blank">Models & Datasets</a><br>
</p>


![https://pages.cs.huji.ac.il/adiyoss-lab/slamming/](../media/slam_web.png)


## Results
We provide some results for our pre-trained models, compared to other SLMs.

| Model                                     | GPUs    | Params | Num Tokens    | sBLIMP ↑  | sStoryCloze ↑ | tStoryCloze ↑ | GenPPL ↓ | Auto-BLEU ↓ |
|-------------------------------------------|---------|--------|---------------|-----------|---------------|---------------|----------|-------------|
| **Speech only pre-training**              |         |        |               |           |               |               |          |             |
| GSLM                                      | 8×V100  | 100M   | 1B            | 54.2      | 53.3          | 66.6          | —        | —           |
| SyllableLM                                | 4×A40   | 300M   | 16B           | 63.7      | —             | 75.4          | —        | —           |
| TWIST-350M                                | 8×V100  | 305M   | 10.8B         | 56.2      | —             | —             | 137.3    | 3.46        |
| TWIST-1.3B                                | 32×V100 | 1B     | 10.8B         | 57.0      | 52.4          | 70.6          | 131.8    | 3.20        |
| TWIST-7B                                  | 32×V100 | 7B     | 36B           | 59.0      | 55.3          | 74.1          | 93.74    | 3.06        |
| TWIST-13B                                 | 32×V100 | 13B    | 36B           | 59.2      | 55.4          | 76.4          | —        | —           |
| Scaled Optimal                            | —       | 823M   | 82B           | **61.3**  | 56.7          | 78.0          | —        | —           |
| Moshi                                     | ?×H100  | 7B     | ?             | 58.9      | **58.7**      | **81.8**      | —        | —           |
| SpiritLM                                  | 64×A100 | 7B     | 100B          | 58.0      | 54.8          | 72.9          | —        | —           |
| **With text / preference optimization**   |         |        |               |           |               |               |          |             |
| Scaling Interleaving                      | —       | 9B     | ~1T           | —         | **62.4**      | 82.9          | —        | —           |
| Moshi                                     | ?×H100  | 7B     | ~720B         | 58.8      | 60.8          | 83.0          | —        | —           |
| SpiritLM                                  | 64×A100 | 7B     | 100B          | 58.3      | 61.0          | 82.9          | —        | —           |
| AlignSLM-1.3B                             | 64×A100 | 1B     | 10.8B + ~158B | 59.8      | 55.0          | 80.0          | —        | —           |
| AlignSLM-7B                               | 64×A100 | 7B     | 36B + ~158B   | **62.3**  | 61.1          | **86.8**      | —        | —           |
| **Ours (_Slam_)**                         |         |        |               |           |               |               |          |             |
| _Slam_ (-DPO)                             | 2×A100  | 358M   | 16.7B         | 58.53     | 58.15         | 80.71         | 67.3     | 3.25        |
| _Slam_                                    | 1×A5000 | 358M   | 1.4B + 5M     | 58.86     | 58.04         | 82.04         | 62.8     | 3.88        |
| _Slam_ (scaled)                           | 2×A100  | 358M   | 16.7B + 9M    | **61.11** | **61.30**     | **84.18**     | **46.6** | 3.75        |


## Citation
If you use this work, please cite our paper:
```bibtex
@misc{maimon2025slamming,
      title={Slamming: Training a Speech Language Model on One GPU in a Day}, 
      author={Gallil Maimon and Avishai Elmakies and Yossi Adi},
      year={2025},
      eprint={2502.15814},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.15814}, 
}
```