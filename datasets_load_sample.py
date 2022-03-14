
from sklearn import datasets
# datasets.load_<TAB>
#
# datasets.fetch_<TAB>
#
# datasets.make_<TAB>

# digits
digits = datasets.load_digits()
digits.keys()

# california_housing
california_housing = datasets.fetch_california_housing()
california_housing.keys()

# gaussian_quantile
gaussian_quantiles = datasets.make_gaussian_quantiles()
type(gaussian_quantiles), len(gaussian_quantiles)
