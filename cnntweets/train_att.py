from cnn_models.preattention_cnn import TextCNNPreAttention



cnn = TextCNNPreAttention(
    sequence_length=60,
    num_classes=3,
    embedding_size=400,
    embedding_size_lex=15,
    num_filters_lex=9,
    filter_sizes=[2],
    num_filters=64,
    l2_reg_lambda=0.1)

