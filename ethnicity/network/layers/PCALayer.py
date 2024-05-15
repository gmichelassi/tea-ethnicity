import tensorflow as tf


class PCALayer(tf.keras.layers.Layer):
    def __init__(self, number_of_components: int = None):
        super(PCALayer, self).__init__()
        self.number_of_components = number_of_components
        self.input_shape = None
        self.components = None

    def build(self, input_shape):
        super(PCALayer, self).build(input_shape)

        self.input_shape = input_shape

        if self.number_of_components is None:
            self.number_of_components = input_shape[-1]

        self.components = self.add_weight(
            name='pca_components',
            shape=(input_shape[-1], self.number_of_components),
            initializer='random_normal',
            trainable=False
        )

    def call(self, inputs):
        """Apply the PCA algorithm to the input tensor.

        Input tensor is reshaped to a 2D tensor, where each row is a sample.
        Steps:
        - Center the data around the origin
        - Compute the covariance matrix
        - Compute the SVD of the covariance matrix to get the eigenvectors
        - Select the top components
        - Project the data onto the top components

        :param inputs:
        :return projected_data:
        """
        batch_size = tf.shape(inputs)[0]
        flattened_inputs = tf.reshape(inputs, (batch_size, -1))

        centered_data = flattened_inputs - tf.reduce_mean(flattened_inputs, axis=0)

        covariance_matrix = tf.matmul(centered_data, centered_data, transpose_a=True) / tf.cast(batch_size, tf.float32)
        covariance_matrix = tf.ensure_shape(covariance_matrix, (self.input_shape[-1], self.input_shape[-1]))

        _, singular_values, eigenvectors = tf.linalg.svd(covariance_matrix)

        top_components = eigenvectors[:, :self.number_of_components]
        self.components.assign(top_components)

        projected_data = tf.matmul(centered_data, top_components)
        output_shape = tf.concat([tf.shape(inputs)[:-1], [self.number_of_components]], axis=0)

        return tf.reshape(projected_data, output_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.number_of_components
