XGBooost
```yaml
model:
  objective: "binary:logistic"
  eval_metric: "logloss"
  tree_method: "hist"
  device: "cuda"
  max_depth: 9
  learning_rate: 0.033837111689684736
  n_estimators: 826
  subsample: 0.896949667796945
  colsample_bytree: 0.659587529547645
  min_child_weight: 1
  gamma: 4.715734894425381
  random_state: 7
  scale_pos_weight: 28
  max_delta_step: 1.730584380845266
  reg_alpha: 7.667845183285583
  reg_lambda: 1.0747337525269672e-06
```

Catboost
```yaml
model:
  loss_function: "Logloss"
  iterations: 850
  depth: 8
  learning_rate: 0.049473263202094754
  l2_leaf_reg: 9.301441348265332
  border_count: 61
  random_strength: 7.92089082642182
  scale_pos_weight: 28
  bootstrap_type: "Bernoulli"
  grow_policy: "Lossguide"
  random_seed: 95
  subsample: 0.8687967802162514
  verbose: 0
  task_type: "GPU"
  cat_features: ["feature_31", "feature_43", "feature_61", "feature_64", "feature_80", "feature_143", "feature_191", "feature_209", "feature_299", "feature_300", "feature_446", "feature_459" ]
```

```yaml
model:
  n_estimators: 127
  learning_rate: 0.012835847658688107
  max_depth: 18
  subsample: 0.7925881420008511
  colsample_bytree: 0.8773532749104792
  reg_alpha: 1.700386005699518
  reg_lambda: 1.5332314072612385
  random_state: 8
```