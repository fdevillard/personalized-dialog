from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
from six.moves import range
from datetime import datetime
import pandas as pd
import random


def zero_nil_slot(t, name=None):
    """
    Overwrites the nil_slot (first row) of the input Tensor with zeros.

    The nil_slot is a dummy slot and should not be trained and influence
    the training algorithm.
    """
    # with tf.op_scope([t], name, "zero_nil_slot") as name:
    with tf.name_scope(name, "zero_nil_slot", [t]) as name:
        t = tf.convert_to_tensor(t, name="t")
        s = tf.shape(t)[1]
        z = tf.zeros(tf.pack([1, s]))
        # z = tf.zeros([1, s])
        return tf.concat(0, [z, tf.slice(t, [1, 0], [-1, -1])], name=name)


def add_gradient_noise(t, stddev=1e-3, name=None):
    """
    Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].

    The input Tensor `t` should be a gradient.

    The output will be `t` + gaussian noise.

    0.001 was said to be a good fixed value for memory networks [2].
    """
    # with tf.op_scope([t, stddev], name, "add_gradient_noise") as name:
    with tf.name_scope(name, "add_gradient_noise", [t, stddev]) as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)


class MemN2NDialog(object):
    """End-To-End Memory Network."""

    MODEL_NAME_SHARED = 'shared'
    MODEL_NAME_SPECIFIC = 'profilespecific'

    def __init__(self,
                 batch_size,
                 vocab_size,
                 candidates_size,
                 sentence_size,
                 embedding_size,
                 candidates_vec,
                 profiles_idx_set,
                 hops=3,
                 max_grad_norm=40.0,
                 nonlin=None,
                 alpha=.5,
                 initializer=tf.random_normal_initializer(stddev=0.1),
                 optimizer=tf.train.AdamOptimizer(learning_rate=1e-2),
                 session=tf.Session(),
                 name='MemN2N',
                 task_id=1,
                 verbose=False):
        """Creates an End-To-End Memory Network

        Args:
            batch_size: The size of the batch.

            vocab_size: The size of the vocabulary (should include the nil word). The nil word
            one-hot encoding should be 0.

            sentence_size: The max size of a sentence in the data. All sentences should be padded
            to this length. If padding is required it should be done with nil one-hot encoding (0).

            candidates_size: The size of candidates

            memory_size: The max size of the memory. Since Tensorflow currently does not support jagged arrays
            all memories must be padded to this length. If padding is required, the extra memories should be
            empty memories; memories filled with the nil word ([0, 0, 0, ......, 0]).

            embedding_size: The size of the word embedding.

            candidates_vec: The numpy array of candidates encoding.

            profiles_idx_set: Set of all possible profile IDs

            hops: The number of hops. A hop consists of reading and addressing a memory slot.
            Defaults to `3`.

            max_grad_norm: Maximum L2 norm clipping value. Defaults to `40.0`.

            nonlin: Non-linearity. Defaults to `None`.

            initializer: Weight initializer. Defaults to `tf.random_normal_initializer(stddev=0.1)`.

            optimizer: Optimizer algorithm used for SGD. Defaults to `tf.train.AdamOptimizer(learning_rate=1e-2)`.

            encoding: A function returning a 2D Tensor (sentence_size, embedding_size). Defaults to `position_encoding`.

            session: Tensorflow Session the model is run with. Defaults to `tf.Session()`.

            name: Name of the End-To-End Memory Network. Defaults to `MemN2N`.
        """

        self._batch_size = batch_size
        self._vocab_size = vocab_size
        self._candidates_size = candidates_size
        self._sentence_size = sentence_size
        self._embedding_size = embedding_size
        self._hops = hops
        self._max_grad_norm = max_grad_norm
        self._alpha = alpha
        self._nonlin = nonlin
        self._init = initializer
        self._opt = optimizer
        self._name = name
        self._candidates = candidates_vec
        self._profile_idx_set = profiles_idx_set
        self._current_profile = None
        self._verbose = verbose

        self._build_inputs()
        self._build_vars()

        print('alpha:', self._alpha)        # TODO Remove me

        # define summary directory
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.root_dir = "%s_%s_%s_%s/" % ('task', str(task_id), 'summary_output', timestamp)
        
        
        # cross entropy
        logits = self._inference(self._profile, self._stories, self._queries) # (batch_size, candidates_size)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, self._answers, name="cross_entropy")
        cross_entropy_sum = tf.reduce_sum(cross_entropy, name="cross_entropy_sum")

        # loss op
        loss_op = cross_entropy_sum

        # gradient pipeline
        grads_and_vars = self._opt.compute_gradients(loss_op)

        def clip_grad(g, v):
            if g is None:
                print("Gradient is indeed none:", v.name)
                return None

            return tf.clip_by_norm(g, self._max_grad_norm), v

        grads_and_vars = [clip_grad(g,v) for g, v in grads_and_vars]
        grads_and_vars = filter(lambda v: v is not None, grads_and_vars)

        nil_grads_and_vars = []
        for g, v in grads_and_vars:
            if v.name in self._nil_vars:
                nil_grads_and_vars.append((zero_nil_slot(g), v))
            else:
                nil_grads_and_vars.append((g, v))

        train_op = self._opt.apply_gradients(nil_grads_and_vars, name="train_op")

        # predict ops
        predict_op = tf.argmax(logits, 1, name="predict_op")
        predict_proba_op = tf.nn.softmax(logits, name="predict_proba_op")
        predict_log_proba_op = tf.log(predict_proba_op, name="predict_log_proba_op")

        # assign ops
        self.loss_op = loss_op
        self.predict_op = predict_op
        self.predict_proba_op = predict_proba_op
        self.predict_log_proba_op = predict_log_proba_op
        self.train_op = train_op
        
        self.graph_output = self.loss_op

        # init_op = tf.initialize_all_variables()
        init_op = tf.global_variables_initializer()
        self._sess = session
        self._sess.run(init_op)

    def _build_inputs(self):
        self._profile = tf.placeholder(tf.int8, shape=(), name="profile")
        self._stories = tf.placeholder(tf.int32, [None, None, self._sentence_size], name="stories")
        self._queries = tf.placeholder(tf.int32, [None, self._sentence_size], name="queries")
        self._answers = tf.placeholder(tf.int32, [None], name="answers")

    def _build_vars(self):
        def build_var_helper():
            nil_word_slot = tf.zeros([1, self._embedding_size])
            A = tf.concat(0, [ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ])
            A = tf.get_variable('A', initializer=A)

            H = tf.get_variable("H", shape=[self._embedding_size, self._embedding_size], initializer=self._init)

            W = tf.concat(0, [ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ])
            W = tf.get_variable("W", initializer=W)

            return {
                'A': A,
                'H': H,
                'W': W,
            }

        with tf.variable_scope(self._name):
            nil_vars = set()

            with tf.variable_scope(MemN2NDialog.MODEL_NAME_SHARED):
                created_vars = build_var_helper()
                nil_vars = nil_vars | {created_vars['A'].name, created_vars['W'].name}

            with tf.variable_scope(MemN2NDialog.MODEL_NAME_SPECIFIC):
                for p in self._profile_idx_set:
                    with tf.variable_scope(str(p)):
                        created_vars = build_var_helper()
                        nil_vars = nil_vars | {created_vars['A'].name, created_vars['W'].name}

        self._nil_vars = nil_vars
        print('nil_vars:', self._nil_vars)

    @staticmethod
    def get_variables():
        variables_names = ["A", "H", "W"]
        return {k: tf.get_variable(k) for k in variables_names}

    @staticmethod
    def get_variables_for_profile(p):
        assert p is not None

        with tf.variable_scope(MemN2NDialog.MODEL_NAME_SPECIFIC, reuse=True):
            with tf.variable_scope(str(p), reuse=True):
                return MemN2NDialog.get_variables()

    def _inference(self, profile, stories, queries):
        def model_inference_helper(A, H, W):
            q_emb = tf.nn.embedding_lookup(A, queries)
            u_0 = tf.reduce_sum(q_emb, 1)
            u = [u_0]
            u_k = u_0       # Typically if self._hops = 0

            for count in range(self._hops):
                m_emb = tf.nn.embedding_lookup(A, stories)
                m = tf.reduce_sum(m_emb, 2)
                # hack to get around no reduce_dot
                u_temp = tf.transpose(tf.expand_dims(u[-1], -1), [0, 2, 1])
                dotted = tf.reduce_sum(m * u_temp, 2)

                # Calculate probabilities
                probs = tf.nn.softmax(dotted)
                # probs = tf.Print(probs, [count, tf.shape(probs), probs], summarize=200)

                probs_temp = tf.transpose(tf.expand_dims(probs, -1), [0, 2, 1])
                c_temp = tf.transpose(m, [0, 2, 1])
                o_k = tf.reduce_sum(c_temp * probs_temp, 2)

                u_k = tf.matmul(u[-1], H) + o_k
                # u_k=u[-1]+tf.matmul(o_k,self.H)

                # nonlinearity
                if self._nonlin:
                    u_k = self._nonlin(u_k)

                u.append(u_k)

            candidates_emb = tf.nn.embedding_lookup(W, self._candidates)
            candidates_emb_sum = tf.reduce_sum(candidates_emb, 1)

            return tf.matmul(u_k, tf.transpose(candidates_emb_sum))

        def construct_model_for_profile(p):
            p_vars = self.get_variables_for_profile(p)
            model = model_inference_helper(**p_vars)

            if self._verbose:
                model = tf.Print(model, [profile], message="Profile {}".format(p))

            return model

        with tf.variable_scope(self._name, reuse=True):
            with tf.variable_scope(MemN2NDialog.MODEL_NAME_SHARED, reuse=True):
                model_vars = self.get_variables()
                shared_result = model_inference_helper(**model_vars)

            def construct_case_element(p):
                return (tf.equal(profile, p), lambda: construct_model_for_profile(p))

            clean_case = [construct_case_element(p) for p in self._profile_idx_set]

            # In tensorflow 0.12, default has to be given (and be a true `constructor`). This behavior is
            # different in more recent implementation of tensorflow.
            # The choice here is to simply add one arbitrary case with a print as default
            def default_constructor():
                first_model = clean_case[0][1]()
                return tf.Print(first_model, data=[tf.constant([0])], message="Called default case in switch. Not good.")

            spec_result = tf.case(clean_case, default=default_constructor, exclusive=True, name='dispatching_profile')

            if self._alpha == 0:
                return shared_result
            elif self._alpha == 1:
                return spec_result
            else:
                specific_scaled = tf.scalar_mul(self._alpha, spec_result)
                shared_scaled = tf.scalar_mul(1-self._alpha, shared_result)
                return tf.add(specific_scaled, shared_scaled)

    @staticmethod
    def _dispatch_arguments_for_profiles(f, batch_size, profiles, stories, *args):
        """
        Helper function that dispatch `f` over same profiles.

        Roughly, this function does the following:
            - Take all indices that share the same profile
            - Call `f` once per profile, and with arguments being the list
              of each list in args that correspond to entry having same profile

        Args:
            f: function to dispatch. Must take as first argument the profile type, and then
               as many arguments as present in `args` (in the same order)
            batch_size: maximum size of the arguments' lists
            profiles: array-like that contains profiles for elements in the batch
            stories: array-like that contains all the stories (will be used to avoid issues where
                     the memory size is different)
            args: values to dispatch (must be of same shape then `profiles`, and have at least one element)

        Returns:
        The list of results with the same index as `profiles`

        Possible improvement:
            - Might be worth to remove the `storie_sizes` computation. This computation comes
              because the parsed data do not have the same story size (there is max size, but
              if the history is shorter, then this size is kept).
              This works sequentially because `data_utils.vectorize_data` takes the minimum
              with respect of the whole batch size (not sure if this was done only to reduce
              slightly the memory footprint)
        """
        assert len(args) > 0, "Must specify at least one argument for f"

        storie_sizes = [s.shape for s in stories]
        df = pd.DataFrame(dict(P=profiles, storie_size=storie_sizes))
        df['ans'] = pd.Series()

        while True:
            not_yet_predicted = df.ans.isnull()
            if not not_yet_predicted.any():
                break

            p = random.choice(df[not_yet_predicted].P.unique())

            first_story_size = df[not_yet_predicted & (df.P == p)].storie_size.iloc[0]
            prediction_slice = df[not_yet_predicted & (df.P == p) & (df.storie_size == first_story_size)].iloc[:batch_size]

            def select_elements(l, indices):
                return np.array([l[i] for i in indices])

            indices = prediction_slice.index

            s = select_elements(stories, indices)
            args_sel = list(map(lambda l: select_elements(l, indices), args))

            preds = f(p, s, *args_sel)

            for i, idx in enumerate(prediction_slice.index):
                df.loc[idx, 'ans'] = preds[i]

        return df.ans.values

    def batch_fit(self, profiles, stories, queries, answers):
        """Runs the training algorithm over the given batch

        Args:
            profiles: Profile numbers (array-like with values in profiles_idx_set)
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)
            answers: Tensor (None, vocab_size)

        Returns:
            loss: floating-point number, the loss computed for the batch
        """
        results = self._dispatch_arguments_for_profiles(self._batch_fit_single_profile,
                                                        self._batch_size,
                                                        profiles,
                                                        stories,
                                                        queries,
                                                        answers)

        return np.mean(results)

    def _batch_fit_single_profile(self, profile, stories, queries, answers):
        """Runs the training algorithm over the given batch

        Args:
            profile: Number in profiles_idx_set, change the model for this profile
                     The stories/queries/answers must be relative to this profile
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)
            answers: Tensor (None, vocab_size)

        Returns:
            loss: floating-point number, the loss computed for the batch
        """
        feed_dict = {self._profile: profile, self._stories: stories, self._queries: queries, self._answers: answers}
        loss, _ = self._sess.run([self.loss_op, self.train_op], feed_dict=feed_dict)

        return [loss] * len(stories)

    def batch_predict(self, profiles, stories, queries):
        """Predicts answers as one-hot encoding.

        Args:
            profiles: Profile numbers (array-like with values in profiles_idx_set)
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)

        Returns:
            answers: Tensor (None, vocab_size)
        """
        results = self._dispatch_arguments_for_profiles(self._predict_single_profile,
                                                        self._batch_size,
                                                        profiles,
                                                        stories,
                                                        queries)

        return results

    def _predict_single_profile(self, profile, stories, queries):
        """Predicts answers as one-hot encoding.

        Args:
            profile: Number in profiles_idx_set, change the model for this profile
                     The stories/queries must be relative to this profile
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)

        Returns:
            answers: Tensor (None, vocab_size)
        """

        feed_dict = {self._profile: profile, self._stories: stories, self._queries: queries}
        return self._sess.run(self.predict_op, feed_dict=feed_dict)
