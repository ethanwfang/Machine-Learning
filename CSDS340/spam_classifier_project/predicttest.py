from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier


def predictTest(trainFeatures, trainLabels, testFeatures):
    impute_estimator = make_pipeline(Nystroem(kernel="polynomial", degree=2, random_state=0), Ridge(alpha=1e3, random_state=0))
    imp3 = [None, []]

    i = IterativeImputer(estimator=impute_estimator, max_iter=10, missing_values=np.nan, random_state=0).fit(trainFeatures)
    imp3[0], imp3[1] = i, i.transform(trainFeatures)

    # train best random forest
    best_params_rf = {
        'criterion': 'gini',
        'max_depth': 20,
        'max_features': 1,
        'min_samples_split': 6,
        'n_estimators': 205
    }
    best_rf = RandomForestClassifier(random_state=0, **best_params_rf).fit(imp3[1], trainLabels)
    predictions_rf = best_rf.predict_proba(imp3[0].transform(testFeatures))[:, 1]

    # train XGBoost
    best_params_xg = {
        'colsample_bytree': 0.932074937042317,
        'gamma': 0.45227982839657344,
        'learning_rate': 0.2976513156416725,
        'max_depth': 3,
        'min_child_weight': 1,
        'n_estimators': 750,
        'reg_alpha': 0.2857460750149036,
        'reg_lambda': 0.21483184008762674,
        'subsample': 1
    }
    best_xgb = XGBClassifier(random_state=0, **best_params_xg).fit(imp3[1], trainLabels)
    predictions_xgb = best_xgb.predict_proba(imp3[0].transform(testFeatures))[:, 1]

    # train stacked classifier
    stack = StackingClassifier(estimators=[('forest', best_rf), ('xgb', best_xgb)], final_estimator=LogisticRegression()).fit(imp3[1], trainLabels)
    return stack.predict_proba(imp3[0].transform(testFeatures))[:, 1]
