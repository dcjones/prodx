
# prodx

This is a basic but flexible Negative-Binomial regression tool intended for
differential expression analysis on single cell count data.

Right now it's mainly a testbed for tinkering with modeling and regressing out
various kind of confounding effects.

Basic usage:
```python

from prodx import DEModel

model = DEModel(
    adata_osn,
    "~ covariate1 * covariate2 + covariate3",
    "~ confounder1 + connfounder2",
)
model.fit()
model.de_results("covariate1[T.True]").to_csv("de-results.csv")
```
