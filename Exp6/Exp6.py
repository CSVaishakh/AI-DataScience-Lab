from sklearn.datasets import load_iris
import pandas as pd

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)

corr_mat = df.corr()
cov_mat = df.cov()

corr_mat.to_csv("Correlation.csv")
cov_mat.to_csv("Covariance.csv")

print(f"Correlation Matrix : \n{corr_mat}\n")
print(f"Corvariance Matrix : \n{cov_mat}\n")