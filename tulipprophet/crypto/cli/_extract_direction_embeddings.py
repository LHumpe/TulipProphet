import os

import click
import tensorflow as tf


@click.command()
@click.option('--model_dir', required=True, type=click.Path(exists=True))
@click.option('--output_dir', required=True, type=click.Path(exists=True))
def _extract_direction_embeddings(model_dir: str, output_dir: str) -> None:
    """
    ClI endpoint to extract embeddings as tsv files.
    Resulting files can be uploaded to and viewed at http://projector.tensorflow.org/.

    Parameters
    ----------
    model_dir : str
        Path to the directory containing the models.
    output_dir : str
        Path for the .tsv files. Directory structure must be created manually.

    Returns
    -------
    None
    """

    # Importing the model
    model = tf.keras.models.load_model(model_dir)

    for artifact in ['short', 'long']:
        # get weights and vocabulary
        weights = model.get_layer(artifact).get_layer('{}_embedding'.format(artifact)).get_weights()[0]
        vocab = model.get_layer(artifact).get_layer('{}_text_vec'.format(artifact)).get_vocabulary()

        # create the paths
        vector_path = os.path.join(output_dir, '{}/vectors.tsv'.format(artifact))
        meta_path = os.path.join(output_dir, '{}/metadata.tsv'.format(artifact))

        # open files
        out_v = open(vector_path, 'w', encoding='utf-8')
        out_m = open(meta_path, 'w', encoding='utf-8')

        # write words and embeddings to file
        for index, word in enumerate(vocab):
            if index == 0:
                continue  # skip 0, it's padding.
            vec = weights[index]
            out_v.write('\t'.join([str(x) for x in vec]) + "\n")
            out_m.write(word + "\n")

        # close files
        out_v.close()
        out_m.close()
