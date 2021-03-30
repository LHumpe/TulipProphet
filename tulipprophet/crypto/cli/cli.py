import click

from ._extract_direction_embeddings import _extract_direction_embeddings
from ._train_direction_lstm import _train_direction_lstm


@click.group()
def cli():
    """
    This function serves as the entry point for the cli and all its sub-commands.
    :return: None
    """
    pass


cli.add_command(_train_direction_lstm, name='lstm_direction_training')
cli.add_command(_extract_direction_embeddings, name='lstm_direction_embeddings')
