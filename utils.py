import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy
import tensorflow as tf
from keras.layers import Dense, Input, LSTM
from keras.models import Model
import pandas as pd
from matplotlib.lines import Line2D
from typing import Iterable

# The 20 amino acids in the dataset
alphabet = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]


def seq2onehot(seq: str, variant: str = None, reverse: bool = False):
    """Converts sequence to one-hot. If variant specified, returns the one-hot for the mutation.
    If reverse, then flips the mutation and wildtype amino acid

    Args:
        seq: (str) Sequence of amino acids
        variant: (str) Variant of wildtype protein, e.g. A2E
        reverse: (bool) Whether to reverse the wildtype/mutant amino acid
    Returns:
        one-hot sequences (List[List]), one-hot mutation (List)
    """

    one_hot = [[0 if aa_type != aa else 1 for aa_type in alphabet] for aa in seq]

    if variant is not None:
        # Get amino acids of the mutation
        mut_idx = int(variant[1:-1]) - 1
        mutant_aa = variant[-1]
        wildtype_aa = variant[0]

        # Use -1 for the wildtype amino acid in the one-hot
        wildtype_idx = alphabet.index(wildtype_aa)
        mutant_idx = alphabet.index(mutant_aa)
        if reverse:
            one_hot[mut_idx][wildtype_idx] = 1
            one_hot[mut_idx][mutant_idx] = -1
        else:
            one_hot[mut_idx][wildtype_idx] = -1

        return one_hot, one_hot[mut_idx]

    return one_hot


class DataGenerator(tf.keras.utils.Sequence):
    """Data generator class to output one hot sequences of proteins

    Attributes:
        df: (pd.DataFrame) Dataframe containing protein sequences and variants
        batch_size: (int) Batch size used for training
        reverse: (bool) Whether to reverse the mutant and wildtype amino acids
        shuffle: (bool) Whether to shuffle the Dataframe
        return_mutant: (bool) Whether to return the mutant amino acid one hot, in addition to the sequence
    """

    def __init__(
        self,
        df: pd.DataFrame,
        batch_size: int,
        reverse: bool = False,
        shuffle: bool = True,
        return_mutant: bool = False,
    ):
        self.df = df
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.return_mutant = return_mutant
        self.reverse = reverse
        if shuffle:
            self.df = self.df.sample(frac=1)

    def on_epoch_end(self):
        """Shuffles the dataframe"""
        if self.shuffle:
            self.df = self.df.sample(frac=1)

    def get_sequences(self, start: int, end: int):
        """Returns the one-hot sequences and mutant of the batch
        Args:
            start: (int) starting index of the batch
            end: (int) ending index of the batch
        Returns:
            one-hot sequences (List[List]), one-hot mutation (List)
        """

        # Get sequences of the batch
        sequences = self.df["aa_seq"][start:end]
        variants = self.df["variant"][start:end]

        # Convert sequences into one-hots
        sequences_onehot = []
        mutations_onehot = []
        for seq, var in zip(sequences, variants):
            seq_onehot, mutant_onehot = seq2onehot(seq, var, self.reverse)
            sequences_onehot.append(seq_onehot)
            mutations_onehot.append(mutant_onehot)

        return sequences_onehot, mutations_onehot

    def __len__(self):
        """Get length of dataframe"""
        return ceil(len(self.df) / self.batch_size)

    def __getitem__(self, idx: int):
        """Get batch of one-hot sequences
        Args:
            idx: (int) Batch index of the whole dataset
        Returns:
            (one-hot sequences (List(List)), one-mutant (List)), ddG score (float)
        """

        # Getting starting and ending idx of the batch in the dataframe
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size

        # Get one-hot sequences
        sequences_onehot, mutations_onehot = self.get_sequences(start, end)
        sequences_onehot = tf.ragged.constant(sequences_onehot)
        mutations_onehot = tf.convert_to_tensor(mutations_onehot)

        # Get ddG scores of the batch
        score = tf.convert_to_tensor(self.df["score"][start:end])
        score = -score if self.reverse else score

        if self.return_mutant:
            return (sequences_onehot, mutations_onehot), score
        else:
            return sequences_onehot, score


def get_true_predicted_energies(
    model: tf.keras.Model,
    test_gen: DataGenerator,
    batch_size: int,
    reverse: bool = False,
    antisym: bool = False,
):
    """Get true and predicted ddG for the test dataset
    Args:
        model: (tf.keras.Model) The model used for inference
        test_gen: (Datagenerator) The data generator containing the test set
        batch_size: (int) Batch size used for faster inference
        reverse: (bool) Whether to reverse the wildtype/mutant
        antisym: (bool) True if using the antisymmeteric LSTM
    Returns:
        true energies (np.ndarray), predicted energies (np.ndarray)"""

    test_gen.reverse = reverse
    test_gen.return_mutant = antisym

    true_energies = []
    pred_energies = []
    for step in tqdm(range(len(test_gen.df) // batch_size)):
        sequence, true_energy = test_gen.__getitem__(step)
        pred_energy = model(sequence)[:, 0]

        pred_energies.append(pred_energy)
        true_energies.append(true_energy)
    pred_energies = np.concatenate(pred_energies)
    true_energies = np.concatenate(true_energies)

    return true_energies, pred_energies


def lstm_model(encoding_dim: int = 100, dropout_rate: float = 0.2):
    """Non anti-symmetric LSTM model
    Args:
        encoding_dim: (int) number of neurons to use for the LSTM cell
        dropout_rate: (float) Drop out rate for LSTM cell
    Return:
        LSTM model (keras.Model)
    """
    # Inputs
    input_seq = Input(shape=(None, len(alphabet)))

    # Layers
    out_seq = LSTM(encoding_dim, dropout=dropout_rate)(input_seq)
    out = Dense(20)(out_seq)
    out = Dense(1, use_bias=False)(out)

    return Model(inputs=input_seq, outputs=out)


def lstm_antisym_model(encoding_dim: int = 100, dropout_rate: float = 0.2):
    """Anti-symmetric LSTM model
    Args:
        encoding_dim: (int) number of neurons to use for the LSTM cell
        dropout_rate: (float) Drop out rate for LSTM cell
    Return:
        anti-symmetric LSTM model (keras.Model)
    """
    # Inputs
    input_seq = Input(shape=(None, len(alphabet)))
    input_mutation = Input(shape=(len(alphabet)))

    # Layers
    out_seq = LSTM(encoding_dim, dropout=dropout_rate)(input_seq)
    out_seq = Dense(20)(out_seq)
    out = input_mutation * out_seq
    out = Dense(1, use_bias=False)(out)

    return Model(inputs=(input_seq, input_mutation), outputs=out)


def compute_rmse(
    true_energies: np.ndarray, pred_energies: np.ndarray, decimals: int = 3
):
    """Computes root mean square error between true_energies and pred_energies"""

    rmse = ((true_energies - pred_energies) ** 2).mean() ** 0.5
    rmse = np.round(rmse, decimals)
    return rmse


def compute_pearsonr(
    true_energies: np.ndarray, pred_energies: np.ndarray, decimals: int = 3
):
    """Computes pearson correlation coefficient between true_energies and pred_energies"""

    pearsonr = scipy.stats.pearsonr(true_energies, pred_energies)
    pearsonr = np.round(pearsonr[0], decimals)
    return pearsonr


def plot_history(history: dict, title: str):
    """Plots the training histories"""

    rmse = np.array(history["loss"]) ** 0.5
    val_rmse = np.array(history["val_loss"]) ** 0.5
    plt.plot(rmse, label="train")
    plt.plot(val_rmse, label="test")
    plt.ylim(
        0,
    )
    plt.legend()
    plt.ylabel("RMSE")
    plt.xlabel("epoch")
    plt.title(title)
    plt.show()


def plot_errors(
    true_energies: np.ndarray,
    pred_energies: np.ndarray,
    rmse: float,
    pearsonr: float,
    true_label: str = "True $\Delta\Delta G$",
    pred_label: str = "Predicted $\Delta\Delta G$",
    savefile: bool = False,
):
    """Scatter plot for the true vs predicted ddGs"""

    plt.figure(figsize=(6, 6))
    plt.rcParams.update({"font.size": 14})

    sc = plt.scatter(true_energies, pred_energies, s=1, alpha=0.2, c="purple")
    plt.plot(np.linspace(-6, 6), np.linspace(-6, 6), "--", c="black")
    plt.xlabel(true_label)
    plt.ylabel(pred_label)
    plt.ylim(-6, 6)
    plt.xlim(-6, 6)
    plt.title(f"RMSE={rmse}    r={pearsonr}")

    legend_elements = [
        Line2D([0], [0], color="black", lw=1, label=true_label, linestyle="--"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=pred_label,
            markerfacecolor="purple",
            markersize=10,
            alpha=0.7,
        ),
    ]
    plt.legend(handles=legend_elements)

    if savefile:
        plt.savefig(
            savefile, bbox_inches="tight", facecolor="white", transparent=False, dpi=150
        )
    plt.show()


def plot_2d_hist(
    x: Iterable[float],
    y: Iterable[float],
    xlabel: str = "$\Delta\Delta G$",
    ylabel: str = "",
    savefile: str = None,
):
    """Plot normalized 2D histograms for ddG"""

    plt.rcParams.update({"font.size": 14})

    # Get histogram of score vs length
    hist, xedges, yedges = np.histogram2d(x, y, bins=25)
    hist = hist[:, :].T

    # Normalize each row by the max
    hist = hist / hist.max(axis=1, keepdims=True)

    # Plot the histogram
    plt.pcolormesh(xedges, yedges, hist)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel("Counts")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if savefile is not None:
        plt.savefig(
            savefile, bbox_inches="tight", facecolor="white", transparent=False, dpi=150
        )
    plt.show()
