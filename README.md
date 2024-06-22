# A Comprehensive Survey of Fairness Evaluation in Computer Vision Tasks
This repository contains the experimental code and the comprehensive survey paper associated with our research.

## Evaluation Framework
In this work, we introduce a novel framework for thoroughly assessing the fairness of visual models.  Our framework extends existing evaluation methods by considering three key dimensions: Inter-Attribute, Inter-Class, and Intra-Class.  These dimensions measure the fairness level of attributes and classes both in relation to each other and internally within an application scenario.  We apply this framework to evaluate and analyze the fairness of various common computer vision models, providing guidance for subsequent fairness optimization efforts.
![avatar](framework.png)


## Surveyed Papers
We have compiled some of the original papers on evaluating metrics in the table below. See the original article for details on the nature of each indicator.
| Metrics | Purpose & Paper |
| --- | --- |
| Demographic Parity | Ensures that the prediction rate is equal across different sensitive groups. [Buolamwini and Gebru (2018)](https://doi.org/10.1145/3287560.3287596) |
| Conditional Statistical Parity | Ensures equal prediction rates across groups after conditioning on certain attributes. [Ramaswamy et al. (2021)](https://arxiv.org/abs/2012.07925) |
| Disparate Impact | Measures the ratio of favorable outcomes between groups. [Barocas and Hardt (2016)](https://arxiv.org/abs/1610.02413) |
| Predictive Parity | Ensures equal positive predictive value across groups. [Chouldechova (2017)](https://doi.org/10.1080/01621459.2017.1324993) |
| Predictive Equality | Ensures equal false positive rates across groups. [Hardt et al. (2016)](https://arxiv.org/abs/1610.02413) |
| Equal Opportunity | Ensures equal true positive rates across groups. [Hardt et al. (2016)](https://arxiv.org/abs/1610.02413) |
| Equalized Odds | Ensures equal true positive and false positive rates across groups. [Hardt et al. (2016)](https://arxiv.org/abs/1610.02413) |
| Conditional Use Accuracy Equality | Ensures equal use accuracy conditioned on the prediction. [Pleiss et al. (2017)](https://arxiv.org/abs/1707.00046) |
| Overall Accuracy Equality | Ensures overall accuracy is equal across groups. [Dieterich et al. (2016)](https://doi.org/10.1109/MCSE.2016.85) |
| Test-fairness | Ensures the test set performance is fair across groups. [Friedler et al. (2019)](https://doi.org/10.1145/3287560.3287583) |
| Well-calibration | Ensures predictions are well-calibrated across groups. [Kleinberg et al. (2017)](https://arxiv.org/abs/1609.05807) |
| Balance for Positive Class | Measures balance for positive class across groups. [Chouldechova (2017)](https://doi.org/10.1080/01621459.2017.1324993) |
| Balance for Negative Class | Measures balance for negative class across groups. [Chouldechova (2017)](https://doi.org/10.1080/01621459.2017.1324993) |
| Tanimoto Coefficient | Measures similarity between predicted and true labels. [Van Deursen et al. (2015)](https://doi.org/10.1016/j.ijforecast.2014.05.004) |
| Cosine Similarity | Measures cosine similarity between predicted and true labels. [Zhang et al. (2017)](https://doi.org/10.1109/TPAMI.2016.2587640) |
| Spearman Correlation | Measures the rank correlation between predicted and true labels. [Spearman (1904)](https://doi.org/10.1037/h0070919) |
| Neuron Distance | Measures the distance between neurons in neural networks. [Li et al. (2015)](https://arxiv.org/abs/1511.07543) |
| Coverage Ratio | Measures the ratio of covered to total instances. [Georgopoulos et al. (2021)](https://arxiv.org/abs/2103.09361) |
| BiasAmpMALS | Measures bias amplification in multi-attribute latent space. [Zhang et al. (2021)](https://arxiv.org/abs/2101.11549) |
| BiasAmp | Measures bias amplification. [Zhang et al. (2018)](https://arxiv.org/abs/1805.07894) |

## Contributing

We encourage contributions from the community. If you encounter any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.