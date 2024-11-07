# alfa-hack-2024
Alfa hack


- Обучение и сохранение моделей (надо делать в сразу в файле train)

Что нужно логировать\унифицировать:
- [x] Препроцессинг данных
- Подбор гиперпараетров модели (точно не нейросетевые \ только деревяные)
- Вывод тестового датасета\валидационного(если нет автоматического логирования той или иной библиотеки)

YAML файл:
- [x] датасет для тренировки (clear-ml data)
- параметры модели \ подбор гиперпараметров(?)
- [x] сохранять модель
- [x] делать ли после тренировки замер метрик и прочего, что будет добавлено кодом


| Модель | Ссылка на эксперимент | Ссылка на ветку | Score (auc-roc) |
|:------:|:---------------------:|:---------------:|:---------------:|
|XGBoost|[ссылка](https://app.clear.ml/projects/d9a3d27351d942639a3a8a5fcd336006/experiments/9acbf0f2089b4593963b027dadf5fc46/hyper-params/configuration/OmegaConf?columns=selected&columns=name&columns=tags&columns=status&columns=project.name&columns=users&columns=started&columns=last_update&columns=type&columns=last_iteration&columns=parent.name&order=-last_update&filter=)| main | 85.335962402764 |
|Blending (CatBoost and XGBoost)|[ссылка](https://app.clear.ml/projects/d9a3d27351d942639a3a8a5fcd336006/experiments/dac20a0966014dbc89c668f21a36657f/artifacts/other/submission/output?columns=selected&columns=name&columns=tags&columns=status&columns=project.name&columns=users&columns=started&columns=last_update&columns=type&columns=last_iteration&columns=parent.name&order=-last_update&filter=)|https://github.com/s1lver29/alfa-hack-2024/tree/blending/models_boosting_and_classic|85.551183484478|