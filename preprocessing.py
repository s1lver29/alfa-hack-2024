import polars as pl
from clearml import Dataset
from sklearn.model_selection import train_test_split


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

    if train_test_split_is:
        train_data, test_data = train_test_split(
            data,
            shuffle=True,
            test_size=0.1,
            stratify=data.select(target),
        )

        return {"train": train_data, "test": test_data}

    return data


def preprocessing_data():
    pass
