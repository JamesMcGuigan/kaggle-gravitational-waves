import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from src.experiments.rnn_starter.CustomDataset import CustomDataset
from src.experiments.rnn_starter.model import build_model


train = pd.read_csv('./input/g2net-gravitational-wave-detection/training_labels.csv')
sub   = pd.read_csv('./input/g2net-gravitational-wave-detection/sample_submission.csv')
train.head()


sample_df = train.sample(frac=1).reset_index(drop=True)

split = int(sample_df.shape[0] * 0.8)
train_df = sample_df[:split]
valid_df = sample_df[split:]

train_dset = CustomDataset(train_df, './input/g2net-n-mels-128-train-images.zip', batch_size=64)
valid_dset = CustomDataset(valid_df, './input/g2net-n-mels-128-train-images.zip', batch_size=64, shuffle=False)
test_dset  = CustomDataset(sub,      './input/g2net-n-mels-128-test-images.zip',  batch_size=64, target=False, shuffle=False)

model = build_model()
model.compile("adam", loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC()])
model.summary()


verbose = True
factor  = 10


train_history = model.fit(
    train_dset,
    use_multiprocessing=True,
    workers=4,
    epochs=10,
    validation_data=valid_dset,
    callbacks=[
        ModelCheckpoint(
            "model_weights.h5",
            save_best_only    = True,
            save_weights_only = True,
        ),
        EarlyStopping(
            monitor  = 'val_loss',
            mode     = 'min',
            verbose  = verbose,
            patience = 10,
            restore_best_weights = True
        ),
        ReduceLROnPlateau(
            monitor  = 'val_loss',
            factor   = 1 / 10,
            patience = 10,
            min_lr   = 1e-5,
            verbose  = True,
        )
    ],
)

model.load_weights('model_weights.h5')

y_pred = model.predict(
    test_dset, use_multiprocessing=True, workers=4, verbose=1
)


sub['target'] = y_pred
sub.to_csv('submission.csv', index=False)
