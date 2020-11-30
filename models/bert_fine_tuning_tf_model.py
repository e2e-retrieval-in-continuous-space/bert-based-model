
import tensorflow_text as text  # Registers the ops.
import tensorflow as tf
import tensorflow_hub as hub

# text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
preprocessor = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/1")
encoder = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3",
    trainable=True)

def encode_batch(sentences):
    encoder_inputs = preprocessor(sentences)  # dict with keys: 'input_mask', 'input_type_ids', 'input_word_ids'
    outputs = encoder(encoder_inputs)
    sequence_output = outputs["sequence_output"]  # [batch_size, seq_length, 768].
    return sequence_output

# out1 = encode_batch(["This is a sample sentence.", "This is another sample sentence."])
# out2 = encode_batch(["This is a sample sentence.", "This is another sample sentence."])

a = 1
# cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])]*sentence1.shape[0]
# input_word_ids = tf.concat([cls, sentence1, sentence2], axis=-1)
# _ = plt.pcolormesh(input_word_ids.to_tensor())