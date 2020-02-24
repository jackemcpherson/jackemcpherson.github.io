Writing a reminder as I always forget how to import the fantastic [fitzRoy R Package](https://github.com/jimmyday12/fitzRoy) into my Python notebook using rpy2.

Here is the working code to do so:

```python
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import 

utils = importr("utils")
install = utils.install_packages

install("fitzRoy")
fitzroy = importr("fitzRoy")
```

We can then use the following to extract player stats for the 2018 - 2019 seasons:

```python
matches = robjects.r.seq(9414, 9721)
player_stats = fitzroy.get_footywire_stats(ids=matches)
```
