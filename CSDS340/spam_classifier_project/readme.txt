Our predictTest.py function requires a few different packages,
but the only one that anaconda doesn't have is XGBoost. 

To install, you can use  "conda install -y xgboost"

And these are all of the *additional* necessary imports needed to run our predictTest.

-- START BLURB --
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
-- END BLURB -- 

Thank you