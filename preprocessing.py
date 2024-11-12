import polars as pl
from clearml import Dataset
from sklearn.model_selection import train_test_split

CAT_FEATURES = [
    "feature_31",
    "feature_43",
    "feature_61",
    "feature_64",
    "feature_80",
    "feature_143",
    "feature_191",
    "feature_209",
    "feature_299",
    "feature_300",
    "feature_446",
    "feature_459",
]


def load_data(
    dataset_project: str,
    dataset_name: str,
    name_file: str,
    train_test_split_is: bool = False,
    target: str = None,
) -> pl.DataFrame | dict[str, pl.DataFrame]:
    """
    Загружает данные из указанного датасета в формате Parquet и логирует информацию о загруженных данных.

    :param dataset_project: Название проекта в ClearML, к которому принадлежит датасет.
    :param dataset_name: Название датасета для загрузки.
    :param name_file: Имя файла в формате Parquet, который нужно загрузить.
    :return: DataFrame с загруженными данными.
    """

    dataset = Dataset.get(
        dataset_name=dataset_name,
        dataset_project=dataset_project,
        writable_copy=False,
    ).get_local_copy()

    data = pl.read_parquet(f"{dataset}/{name_file}")
    if data["smpl"].head(1)[0] == "train":
        data = data.drop("id")
    data = data.drop("smpl")

    data_sub_train_positive = data.filter(pl.col("target") == 1)
    data_sub_train_negative = data.filter(pl.col("target") == 0).sample(
        fraction=0.3, seed=421
    )
    data = pl.concat([data_sub_train_positive, data_sub_train_negative])

    # ------
    data = preprocessing_data(data)
    # ------

    if train_test_split_is:
        train_data, test_data = train_test_split(
            data,
            shuffle=True,
            test_size=0.1,
            stratify=data.select(target),
        )

        return {"train": train_data, "test": test_data}

    return data


def preprocessing_data(data: pl.DataFrame):
    """Преобразуем типы данных для catboost в integer"""
    for column in CAT_FEATURES:
        data = data.with_columns(
            data[column].cast(pl.Int64, strict=False).alias(column)
        )

    return data
