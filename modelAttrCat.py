from modules import *


class Model():
    def __init__(self, usernum, itemnum, args, reuse=None):
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.u = tf.placeholder(tf.int32, shape=(None))
        self.input_seq = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.pos = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.neg = tf.placeholder(tf.int32, shape=(None, args.maxlen))

        self.seq_attr = tf.placeholder(tf.float32, shape=(None, args.maxlen, 1128))
        self.pos_attr = tf.placeholder(tf.float32, shape=(None, args.maxlen, 1128))
        self.neg_attr = tf.placeholder(tf.float32, shape=(None, args.maxlen, 1128))
        self.test_attr = tf.placeholder(tf.float32, shape=(101, 1128))

        pos = self.pos
        neg = self.neg
        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq, 0)), -1)

        '''
        tf.keras.layers.Dense(
            units, activation=None, use_bias=True, kernel_initializer='glorot_uniform',
            bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
            activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
            **kwargs
        )
        '''
        l2_coeff = 0.01
        with tf.variable_scope("SASRec", reuse=reuse):
            # Attribute encoding
            with tf.variable_scope("attr_enc", reuse=reuse):
                seq_attr_enc = tf.layers.dense(tf.reshape(self.seq_attr, [-1, 1128]),
                                               args.hidden_units,
                                               kernel_regularizer=regularizers.l2(l2_coeff), bias_regularizer=regularizers.l2(l2_coeff))
                self.seq_attr_enc = tf.reshape(seq_attr_enc, [-1, args.maxlen, args.hidden_units])

            with tf.variable_scope("attr_enc", reuse=True):
                pos_attr_enc = tf.layers.dense(tf.reshape(self.pos_attr, [-1, 1128]),
                                               args.hidden_units,
                                               kernel_regularizer=regularizers.l2(l2_coeff), bias_regularizer=regularizers.l2(l2_coeff))
                # self.pos_attr_enc = tf.reshape(pos_attr_enc, [-1, args.maxlen, args.hidden_units])

            with tf.variable_scope("attr_enc", reuse=True):
                neg_attr_enc = tf.layers.dense(tf.reshape(self.neg_attr, [-1, 1128]),
                                               args.hidden_units,
                                               kernel_regularizer=regularizers.l2(l2_coeff), bias_regularizer=regularizers.l2(l2_coeff))
                # self.neg_attr_enc = tf.reshape(neg_attr_enc, [-1, args.maxlen, args.hidden_units])

            with tf.variable_scope("attr_enc", reuse=True):
                test_attr_enc = tf.layers.dense(self.test_attr,
                                                args.hidden_units,
                                                kernel_regularizer=regularizers.l2(l2_coeff), bias_regularizer=regularizers.l2(l2_coeff))
                # self.neg_attr_enc = tf.reshape(neg_attr_enc, [-1, args.maxlen, args.hidden_units])

            # sequence embedding, item embedding table
            self.seq, item_emb_table = embedding(self.input_seq,
                                                 vocab_size=itemnum + 1,
                                                 num_units=args.hidden_units,
                                                 zero_pad=True,
                                                 scale=True,
                                                 l2_reg=args.l2_emb,
                                                 scope="input_embeddings",
                                                 with_t=True,
                                                 reuse=reuse
                                                 )

            # Positional Encoding
            t, pos_emb_table = embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]),
                vocab_size=args.maxlen,
                num_units=args.hidden_units*2,
                zero_pad=False,
                scale=False,
                l2_reg=args.l2_emb,
                scope="dec_pos",
                reuse=reuse,
                with_t=True
            )
            # Option #1: Position encoding only on moive embedding
            # self.seq += t
            # self.seq = tf.concat([self.seq, self.seq_attr_enc], axis=-1)

            # Option #2: Position encoding on both moive embedding and attr encoding
            self.seq = tf.concat([self.seq, self.seq_attr_enc], axis=-1)
            self.seq += t


            # Dropout
            self.seq = tf.layers.dropout(self.seq,
                                         rate=args.dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))
            self.seq *= mask

            # Build blocks

            for i in range(args.num_blocks):
                with tf.variable_scope("num_blocks_%d" % i):

                    # Self-attention
                    self.seq = multihead_attention(queries=normalize(self.seq),
                                                   keys=self.seq,
                                                   num_units=args.hidden_units*2,
                                                   num_heads=args.num_heads,
                                                   dropout_rate=args.dropout_rate,
                                                   is_training=self.is_training,
                                                   causality=True,
                                                   scope="self_attention")

                    # Feed forward
                    self.seq = feedforward(normalize(self.seq), num_units=[args.hidden_units*2, args.hidden_units*2],
                                           dropout_rate=args.dropout_rate, is_training=self.is_training)
                    self.seq *= mask

            self.seq = normalize(self.seq)

        pos = tf.reshape(pos, [tf.shape(self.input_seq)[0] * args.maxlen])
        neg = tf.reshape(neg, [tf.shape(self.input_seq)[0] * args.maxlen])
        pos_emb = tf.concat([tf.nn.embedding_lookup(item_emb_table, pos), pos_attr_enc], axis=-1)
        neg_emb = tf.concat([tf.nn.embedding_lookup(item_emb_table, neg), neg_attr_enc], axis=-1)
        seq_emb = tf.reshape(self.seq, [tf.shape(self.input_seq)[0] * args.maxlen, args.hidden_units*2])

        self.test_item = tf.placeholder(tf.int32, shape=(101))
        test_item_emb = tf.concat([tf.nn.embedding_lookup(item_emb_table, self.test_item), test_attr_enc], axis=-1)
        self.test_logits = tf.matmul(seq_emb, tf.transpose(test_item_emb))
        # import pdb; pdb.set_trace()
        self.test_logits = tf.reshape(self.test_logits, [tf.shape(self.input_seq)[0], args.maxlen, 101])
        # import pdb; pdb.set_trace()
        self.test_logits = self.test_logits[:, -1, :]
        # import pdb; pdb.set_trace()

        # prediction layer
        self.pos_logits = tf.reduce_sum(pos_emb * seq_emb, -1)
        self.neg_logits = tf.reduce_sum(neg_emb * seq_emb, -1)

        # ignore padding items (0)
        istarget = tf.reshape(tf.to_float(tf.not_equal(pos, 0)), [tf.shape(self.input_seq)[0] * args.maxlen])
        self.loss = tf.reduce_sum(
            - tf.log(tf.sigmoid(self.pos_logits) + 1e-24) * istarget -
            tf.log(1 - tf.sigmoid(self.neg_logits) + 1e-24) * istarget
        ) / tf.reduce_sum(istarget)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss += sum(reg_losses)

        tf.summary.scalar('loss', self.loss)
        self.auc = tf.reduce_sum(
            ((tf.sign(self.pos_logits - self.neg_logits) + 1) / 2) * istarget
        ) / tf.reduce_sum(istarget)

        if reuse is None:
            tf.summary.scalar('auc', self.auc)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, beta2=0.98)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        else:
            tf.summary.scalar('test_auc', self.auc)

        self.merged = tf.summary.merge_all()

    def predict(self, sess, u, seq, item_idx, seq_attr, item_idx_attr):
        return sess.run(self.test_logits,
                        {self.u: u, self.input_seq: seq, self.test_item: item_idx,
                         self.seq_attr: seq_attr, self.test_attr: item_idx_attr,
                         self.is_training: False})
