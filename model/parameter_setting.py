def learning_setting(variable_name):
    if 'excEcon' in variable_name:
        RidgeSetting = {
            'alpha' : 0.98,
            'fit_intercept' : False,
            'Lags' : 13
        }
        LGBMSetting = {
            'min_child_samples' : 25,
            'max_depth' : 8,
            'colsample_bytree' : 0.4,
            'boosting_type' : 'dart',
            'Lags' : 26
        }
        RandomForestSetting = {
            'n_estimators' : 800,
            'min_samples_leaf' : 11,
            'max_features' : 'sqrt',
            'max_depth' : 8,
            'Lags' : 52
        }
    elif 'excFdmt' in variable_name:
        RidgeSetting = {
            'alpha' : 0.98,
            'fit_intercept' : False,
            'Lags' : 52
        }
        LGBMSetting = {
            'min_child_samples' : 17,
            'max_depth' : 3,
            'colsample_bytree' : 0.9,
            'boosting_type' : 'dart',
            'Lags' : 52
        }
        RandomForestSetting = {
            'n_estimators' : 600,
            'min_samples_leaf' : 24,
            'max_features' : 'sqrt',
            'max_depth' : 11,
            'Lags' : 52
        }
    elif 'excSent' in variable_name:
        RidgeSetting = {
            'alpha' : 0.98,
            'fit_intercept' : False,
            'Lags' : 13
        }
        LGBMSetting = {
            'min_child_samples' : 25,
            'max_depth' : 8,
            'colsample_bytree' : 0.4,
            'boosting_type' : 'dart',
            'Lags' : 52
        }
        RandomForestSetting = {
            'n_estimators' : 700,
            'min_samples_leaf' : 12,
            'max_features' : 1.0,
            'max_depth' : 2,
            'Lags' : 13
        }
    elif 'onlyEcon' in variable_name:
        RidgeSetting = {
            'alpha' : 0.98,
            'fit_intercept' : False,
            'Lags' : 13
        }
        LGBMSetting = {
            'min_child_samples' : 17,
            'max_depth' : 3,
            'colsample_bytree' : 0.9,
            'boosting_type' : 'dart',
            'Lags' : 52
        }
        RandomForestSetting = {
            'n_estimators' : 600,
            'min_samples_leaf' : 24,
            'max_features' : 'sqrt',
            'max_depth' : 11,
            'Lags' : 52
        }
    elif 'onlyFdmt' in variable_name:
        RidgeSetting = {
            'alpha' : 0.98,
            'fit_intercept' : False,
            'Lags' : 13
        }
        LGBMSetting = {
            'min_child_samples' : 30,
            'max_depth' : 7,
            'colsample_bytree' : 0.4,
            'boosting_type' : 'dart',
            'Lags' : 52
        }
        RandomForestSetting = {
            'n_estimators' : 600,
            'min_samples_leaf' : 31,
            'max_features' : 'sqrt',
            'max_depth' : 3,
            'Lags' : 52
        }
    else:
        RidgeSetting = {
            'alpha' : 0.98,
            'fit_intercept' : False,
            'Lags' : 13
        }
        LGBMSetting = {
            'min_child_samples' : 12,
            'max_depth' : 11,
            'colsample_bytree' : 0.1,
            'boosting_type' : 'dart',
            'Lags' : 52
        }
        RandomForestSetting = {
            'n_estimators' : 300,
            'min_samples_leaf' : 12,
            'max_features' : 1.0,
            'max_depth' : 11,
            'Lags' : 26
        }

    setting = {'RidgeSetting': RidgeSetting,
               'LGBMSetting': LGBMSetting,
               'RandomForestSetting': RandomForestSetting}

    return setting
