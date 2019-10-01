import tensorflow as tf
from models import rmc_vanilla, rmc_att, rmc_vdcnn, lstm_vanilla, rmc_att_topic, amazon_attr, customer_reviews

generator_dict = {
    'lstm_vanilla': lstm_vanilla.generator,
    'rmc_vanilla': rmc_vanilla.generator,
    'rmc_att': rmc_att.generator,
    'rmc_att_topic': rmc_att_topic.generator,
    'rmc_vdcnn': rmc_vdcnn.generator,
    'amazon_attribute': amazon_attr.generator,
    'CustomerReviews': customer_reviews.ReviewGenerator
}

discriminator_dict = {
    'lstm_vanilla': lstm_vanilla.discriminator,
    'rmc_vanilla': rmc_vanilla.discriminator,
    'rmc_att': rmc_att.discriminator,
    'rmc_att_topic': rmc_att_topic.discriminator,
    'rmc_vdcnn': rmc_vdcnn.discriminator,
    'amazon_attribute': amazon_attr.discriminator,
    'CustomerReviews': customer_reviews.ReviewDiscriminator
}

discriminator_topic_dict = {
    'standard': rmc_att_topic.topic_discriminator,
    'reuse_att_topic': rmc_att_topic.topic_discriminator_reuse
}


def get_generator(model_name, scope='generator', **kwargs):
    model_func = generator_dict[model_name]
    return tf.make_template(scope, model_func, **kwargs)


def get_discriminator(model_name, scope='discriminator', **kwargs):
    model_func = discriminator_dict[model_name]
    return tf.make_template(scope, model_func, **kwargs)


def get_topic_discriminator(model_name, scope='topic_discriminator', **kwargs):
    model_func = discriminator_topic_dict[model_name]
    return tf.make_template(scope, model_func, **kwargs)
