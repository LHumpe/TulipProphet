import os

import click
import tensorflow as tf


@click.command()
@click.option('--model_dir', required=True, type=click.Path(exists=True))
@click.option('--output_dir', required=True, type=click.Path(exists=True))
def _extract_direction_embeddings(model_dir: str, output_dir: str):
    model = tf.keras.models.load_model(model_dir)
    for artifact in ['short', 'long']:
        weights = model.get_layer(artifact).get_layer('{}_embedding'.format(artifact)).get_weights()[0]
        vocab = model.get_layer(artifact).get_layer('{}_text_vec'.format(artifact)).get_vocabulary()

        vector_path = os.path.join(output_dir, '{}/vectors.tsv'.format(artifact))
        meta_path = os.path.join(output_dir, '{}/metadata.tsv'.format(artifact))
        click.echo(meta_path)

        out_v = open(vector_path, 'w', encoding='utf-8')
        out_m = open(meta_path, 'w', encoding='utf-8')

        for index, word in enumerate(vocab):
            if index == 0:
                continue  # skip 0, it's padding.
            vec = weights[index]
            out_v.write('\t'.join([str(x) for x in vec]) + "\n")
            out_m.write(word + "\n")

        out_v.close()
        out_m.close()
