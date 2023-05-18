# Environment
python==3.8.15
***
pytorch==1.13.0

scikit-learn==1.1.3

matplotlib==3.5.1

PyG

# Usage

```python
conda create -n miao python=3.8.15
conda activate miao
pip install -r requirements.txt
conda install pyg -c pyg -c conda-forge
python cora.py
python citeseer.py
python pubmed.py
```

# Result
| Dataset | Cora | Citeseer | Pubmed |
|:-: | :-: | :-: | :-: |
|Accuracy|0.8020|0.6690|0.7880|
|F1_score|0.7926|0.6480|0.7843|
|AUC|0.96|0.90|0.91|
