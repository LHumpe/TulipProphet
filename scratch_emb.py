import tensorflow as tf
import io

model = tf.keras.models.load_model('/mnt/ShareDrive/Development/Privat/TulipArena/singles/TulipProphet/tulipprophet/weights/crypto_model')
weights = model.get_layer('Body').get_layer('bdy_embedding').get_weights()[0]
vocab = model.get_layer('Body').get_layer('text_vectorization').get_vocabulary()

out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
out_m = io.open('metadata.tsv', 'w', encoding='utf-8')

for index, word in enumerate(vocab):
  if  index == 0: continue # skip 0, it's padding.
  vec = weights[index]
  out_v.write('\t'.join([str(x) for x in vec]) + "\n")
  out_m.write(word + "\n")
out_v.close()
out_m.close()