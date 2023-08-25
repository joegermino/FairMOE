# FairMOE
Git Repository for "Fairness-aware Mixture of Experts with Interpretability Budgets"


DataLoader.py: methods used to load and encode the datasets

FairnessMetrics.py: methods used to calculate SP and EO

FairMOE.py: code for proposal, FairMOE

expertiments.py: code for experiments to test FairMOE against baselines across 7 datasets

results: results from 10 runs of experiments.py

baselines: code for benchmarks which FairMOE was compared against

data: publicly available datasets used in experiments (courtesy UCI Machine Learning Repository)

analysis.ipynb: Initial analysis of results from experiments

# Benchmark References:

[1] Agarwal, A., Beygelzimer, A., Dudik, M., Langford, J. & Wallach, H.. (2018). A Reductions Approach to Fair Classification. Proceedings of the 35th International Conference on Machine Learning, in Proceedings of Machine Learning Research 80:60-69 Available from https://proceedings.mlr.press/v80/agarwal18a.html.

[2] Hardt, Moritz, Eric Price, and Nati Srebro. "Equality of opportunity in supervised learning." Advances in neural information processing systems 29 (2016).

[3] Kewen Peng, Joymallya Chakraborty, and Tim Menzies. 2022. FairMask: Better Fairness via Model-Based Rebalancing of Protected Attributes. IEEE Trans. Softw. Eng. 49, 4 (April 2023), 2426–2439. https://doi.org/10.1109/TSE.2022.3220713

[4] Zafar, M.B., Valera, I., Rogriguez, M.G. & Gummadi, K.P.. (2017). Fairness Constraints: Mechanisms for Fair Classification.  Proceedings of the 20th International Conference on Artificial Intelligence and Statistics, in Proceedings of Machine Learning Research 54:962-970 Available from https://proceedings.mlr.press/v54/zafar17a.html.

# Data Sources:

Becker,Barry and Kohavi,Ronny. (1996). Adult. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20.

Hofmann,Hans. (1994). Statlog (German Credit Data). UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77.

Kuzilek J., Hlosta M., Zdrahal Z. Open University Learning Analytics dataset Sci. Data 4:170171 doi: 10.1038/sdata.2017.171 (2017).

Van der Laan, P. (2000). The 2001 census in the netherlands. In Conference the Census of Population

Le Quy, T., Roy, A., Friege, G., & Ntoutsi, E. (2021). Fair-capacitated clustering. In Proceedings of the 14th International Conference on Educational Data Mining (EDM21). (pp. 407-414).

Moro,S., Rita,P., and Cortez,P.. (2012). Bank Marketing. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306.

Wightman, L. F. (1998). LSAC national longitudinal bar passage study. LSAC research report series.

Yeh,I-Cheng. (2016). default of credit card clients. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H.
