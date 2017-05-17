import pandas as pd
import numpy as np 
import tensorflow as tf
from collections import namedtuple

class CoxPH:

    def __init__(self, df, duration_col, event_col):
        self.df = df.copy()
        self.df.sort_values(by=duration_col, inplace=True)
        self.duration_col = duration_col
        self.event_col = event_col
        self.preprocess()

    def preprocess(self):
        self.E = self.df[self.event_col]
        del self.df[self.event_col]
        self.T = self.df[self.duration_col]
        del self.df[self.duration_col]
        self.E = self.E.astype(bool)

    @staticmethod
    def update_step(loop):
        
        addition_to_running_total = tf.exp(tf.multiply(x[i] , b))

        return Loop(
                tf.add(loop.i, 1),
                addition_to_running_total,
                tf.concat([loop.built_array, look.running_total + addition_to_running_total], 0)
               )

    def fit(self):
        
        b = tf.Variable(3., tf.float32)

        Loop = namedtuple("Loop", ["i", "running_total", "built_array"])
        while_condition = lambda x: x.i < 48
        
        starting_counter = tf.constant(1)
        starting_running_total = tf.multiply(2.3, b)
        starting_built_array = tf.concat([[starting_running_total]], 0)
        starting_point = Loop(starting_counter, starting_running_total, starting_built_array)

        loop_outcome = tf.while_loop(while_condition, self.update_step, starting_point, shape_invariants=(i.get_shape(), x.get_shape(), tf.TensorShape([None])))

        log_bigger_equal = tf.log(loop_outcome.built_array)

        indiv_coeff = tf.multiply(x, b)

        log_liks = tf.subtract(indiv_coeff, log_bigger_equal)
        log_lik_sum = tf.reduce_sum(log_liks)

        loss = tf.multiply(log_lik_sum, tf.constant(-1.0))
        optimizer = tf.train.GradientDescentOptimizer(0.01)
        train = optimizer.minimize(loss)

        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(10000):
            sess.run(train,  {x:self.df.T})

        out = sess.run([b], {x:df.age.astype(float)})
        return out.b

from lifelines.datasets import load_rossi
rossi_dataset = load_rossi()
cf = CoxPH(rossi_dataset, "week", "arrest")
cf.fit()
print(cf.hazards_)
