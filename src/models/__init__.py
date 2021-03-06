from tensorflow import make_template

from models import rmc_vanilla, rmc_att, rmc_vdcnn, lstm_free, lstm_topic, rmc_att_topic, amazon_attr, customer_reviews

generator_dict = {
    'lstm_topic': lstm_topic.generator,
    'lstm_free': lstm_free.generator,
    'rmc_vanilla': rmc_vanilla.generator,
    'rmc_att': rmc_att.generator,
    'rmc_att_topic': rmc_att_topic.generator,
    'rmc_vdcnn': rmc_vdcnn.generator,
    'amazon_attribute': amazon_attr.AmazonGenerator,
    'CustomerReviews': customer_reviews.ReviewGenerator
}

discriminator_dict = {
    'rmc_vanilla': rmc_vanilla.discriminator,
    'rmc_att': rmc_att.discriminator,
    'rmc_att_topic': rmc_att_topic.discriminator,
    'rmc_vdcnn': rmc_vdcnn.discriminator,
    'amazon_attribute': amazon_attr.AmazonDiscriminator,
    'CustomerReviews': customer_reviews.ReviewDiscriminator
}

discriminator_topic_dict = {
    'standard': rmc_att_topic.topic_discriminator,
    'reuse_att_topic': rmc_att_topic.topic_discriminator_reuse,
    'CustomerReviews': customer_reviews.ReviewDiscriminator
}


def get_generator(model_name, scope='generator', **kwargs):
    model_func = generator_dict[model_name]
    return make_template(scope, model_func, **kwargs)


def get_discriminator(model_name, scope='discriminator', **kwargs):
    model_func = discriminator_dict[model_name]
    return make_template(scope, model_func, **kwargs)


def get_topic_discriminator(model_name, scope='topic_discriminator', **kwargs):
    model_func = discriminator_topic_dict[model_name]
    return make_template(scope, model_func, **kwargs)
