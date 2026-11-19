import pandas as pd


def test_dataset():
    df = pd.read_csv("data/tetris_perfect_clear_sequences.csv")

    # Check the number of sequences
    assert len(df) == 840

    # Check the number of successful sequences
    assert (df["success"] == 1).sum() == 711

    # Check the number of unsuccessful sequences
    assert (df["success"] == 0).sum() == 129

    # Check for missing values
    assert not df.isnull().values.any()

    # Check for duplicate sequences
    assert not df.duplicated(
        subset=["piece_1", "piece_2", "piece_3", "piece_4"]
    ).any()
