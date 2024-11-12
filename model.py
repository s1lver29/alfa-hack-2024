from pytorch_tabular import TabularModel
from pytorch_tabular.models import FTTransformerConfig
from pytorch_tabular.config import (
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
)
from pytorch_tabular.models.common.heads import LinearHeadConfig
import polars as pl


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


def train_model(data: pl.DataFrame, target: pl.DataFrame, **args_model):
    num_col_names = [column for column in data.columns if column not in CAT_FEATURES]
    cat_col_names = CAT_FEATURES

    data = data.with_columns(target["target"])

    data_config = DataConfig(
        target=[
            "target"
        ],  # target should always be a list. Multi-targets are only supported for regression. Multi-Task Classification is not implemented
        continuous_cols=num_col_names,
        categorical_cols=cat_col_names,
    )

    trainer_config = TrainerConfig(
        auto_lr_find=True,  # Runs the LRFinder to automatically derive a learning rate
        batch_size=128,
        max_epochs=20,
        # early_stopping="valid_loss",  # Monitor valid_loss for early stopping
        # early_stopping_mode="min",  # Set the mode as min because for val_loss, lower is better
        # early_stopping_patience=5,  # No. of epochs of degradation training will wait before terminating
        # checkpoints="valid_loss",  # Save best checkpoint monitoring val_loss
        # load_best=True,  # After training, load the best checkpoint
        #     accelerator="cpu"
    )

    optimizer_config = OptimizerConfig()

    head_config = LinearHeadConfig(
        layers="",  # No additional layer in head, just a mapping layer to output_dim
        dropout=0.1,
        initialization="kaiming",
    ).__dict__

    model_config = FTTransformerConfig(
        task="classification",
        num_attn_blocks=6,
        num_heads=4,
        learning_rate=1e-5,
        head="LinearHead",  # Linear Head
        head_config=head_config,  # Linear Head Config
        metrics=["f1_score", "accuracy"],
        metrics_params=[{"num_classes": 2}, {}],  # f1_score needs num_classes
        metrics_prob_input=[True, False],
    )

    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
        verbose=True,
    )

    tabular_model.fit(train=data.to_pandas(), validation=None)

    return tabular_model


def predict(model, data: pl.DataFrame, predict_proba_is: bool = True):
    print(model.predict(data.to_pandas()))
    if predict_proba_is:
        return model.predict(data.to_pandas())["1_probability"].values

    return model.predict(data.to_pandas())["prediction"].values
