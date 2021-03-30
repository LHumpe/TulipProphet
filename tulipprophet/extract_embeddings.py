import io

import tensorflow as tf

if __name__ == '__main__':
    version = 'V1'
    model = tf.keras.models.load_model('../local_output/models/{}'.format(version))
    for artifact in ['short', 'long']:
        weights = model.get_layer(artifact).get_layer('{}_embedding'.format(artifact)).get_weights()[0]
        vocab = model.get_layer(artifact).get_layer('{}_text_vec'.format(artifact)).get_vocabulary()

        out_v = io.open('../local_output/embeddings/{}/{}/vectors.tsv'.format(version, artifact), 'w', encoding='utf-8')
        out_m = io.open('../local_output/embeddings/{}/{}/metadata.tsv'.format(version, artifact), 'w',
                        encoding='utf-8')

        for index, word in enumerate(vocab):
            if index == 0: continue  # skip 0, it's padding.
            vec = weights[index]
            out_v.write('\t'.join([str(x) for x in vec]) + "\n")
            out_m.write(word + "\n")

        out_v.close()
        out_m.close()
