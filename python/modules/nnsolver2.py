import tensorflow as tf
from modules import utility as ut
import scipy.stats as ss
import tables
import os
import numpy as np
from tensorflow.python.training.tracking.data_structures import NoDependency
import pandas as pd


class Domain:
    def __init__(self, dtype, *args):
        self.dtype = dtype
        self.dim = int(len(args)/2)
        self.intervals = [args[2*d] for d in range(self.dim)]
        self.distributions = [args[2*d + 1] for d in range(self.dim)]

    def sample(self, num_samples, **params):
        samples = []
        for d in range(self.dim):
            if self.distributions[d] is 'uniform':
                samples.append(tf.random.uniform(shape=(num_samples, 1), minval=self.intervals[d][0], maxval=self.intervals[d][1], dtype=self.dtype))
            elif self.distributions[d] is 'truncated_normal':
                rvs = ss.truncnorm.rvs(self.intervals[d][0],  self.intervals[d][1], size=(num_samples, 1), **params)
                samples.append(tf.convert_to_tensor(rvs, dtype=self.dtype))
        self.sample_size =  num_samples
        return samples

class DataPipelineH5:
    def __init__(self, db_path, dtype=tf.float32):
        self.db_path = db_path
        self.dtype = dtype
        self.domains = []
        if not os.path.isfile(db_path):
            hdf5 = tables.open_file(db_path, 'w')
            hdf5.close()

    def add_domain(self, *args):
        self.domains.append(Domain(self.dtype, *args))

    def build_db(self, num_pts=None, chunk_size=int(1e5), normalize=True):
        if num_pts is None:
            num_pts = [1000] * len(self.domains)
        hdf5 = tables.open_file(self.db_path, 'a')
        col = tables.Float32Col if self.dtype == tf.float32 else tables.Float64Col
        for i, domain in enumerate(self.domains):
            point_description = {}
            for j in range(domain.dim):
                point_description['x' + str(j)] = col(pos = j)
            try:
                tbl = hdf5.create_table(hdf5.root, 'domain_' + str(i), point_description)
                for j in range(int(num_pts[i]/chunk_size) if chunk_size < num_pts[i] else 1):
                    data = domain.sample(chunk_size if chunk_size < num_pts[i] else num_pts[i])
                    if normalize:
                        for d in range(domain.dim):
                            a, b = domain.intervals[d]
                            data[d] = (data[d] - a) / (b - a)
                    tbl.append(tf.concat(data, axis=1).numpy())
                    tbl.flush()
                    print('Chunk #{} of domain #{} has been written.'.format(j, i))
            except:
                self.domains[i].sample_size = getattr(hdf5.root, 'domain_' + str(i)).nrows
        hdf5.close()

    def open_db(self):
        self.db = tables.open_file(self.db_path, 'r')

    def close_db(self):
        try:
            self.db.close()
        except:
            pass

    def read_db(self, num_pts=None, start=None):
        if num_pts is None:
            num_pts = [int(1e3)] * len(self.domains)
        if start is None:
            start = [0] * len(self.domains)
        samples = []
        for i, domain in enumerate(self.domains):
            samples.append(tf.convert_to_tensor(getattr(self.db.root, 'domain_' + str(i))\
                                                .read(start=start[i], stop=start[i] + num_pts[i]).tolist(), dtype=self.dtype))
        return samples

class DataPipelineCSV:
    def __init__(self, db_path, dtype=tf.float32):
        self.db_path = db_path
        self.dtype = dtype
        self.domains = []

    def add_domain(self, *args):
        self.domains.append(Domain(self.dtype, *args))

    def build_db_separate(self, num_pts, chunk_size=int(1e6), normalize=False):
        columns = {}
        for i, domain in enumerate(self.domains):
            for j in range(int(num_pts[i]/chunk_size) if chunk_size < num_pts[i] else 1):
                data = domain.sample(chunk_size if chunk_size < num_pts[i] else num_pts[i])
                if normalize:
                    for d in range(domain.dim):
                        a, b = domain.intervals[d]
                        data[d] = (data[d] - a) / (b - a)
                for d in range(domain.dim):
                    columns['x' + str(d)] = data[d].numpy().flatten()
                if j==0:
                    columns = ['x' + str(d) for d in range(domain.dim)]
                    pd.DataFrame(columns, columns=list(columns)).to_csv(self.db_path + '/domain_{}.csv'.format(i), header=False, index=False)
                else:
                    with open(self.db_path + '/domain_{}.csv'.format(i), 'a', newline='') as csv_file:
                        pd.DataFrame(point_description).to_csv(csv_file, header=False, index=False)
                    csv_file.close()
                print('Chunk #{} of domain #{} has been written.'.format(j, i))
                self.domains[i].sample_size = int(num_pts[i]/chunk_size)*chunk_size if chunk_size < num_pts[i] else num_pts[i]

    def build_db_together(self, num_pts=1000, chunk_size=int(1e6), normalize=False):
        columns = {}
        combined_dim = sum([d.dim for d in self.domains])
        for j in range(int(num_pts/chunk_size) if chunk_size < num_pts else 1):
            combined_points = []
            for i, domain in enumerate(self.domains):
                data = domain.sample(chunk_size if chunk_size < num_pts else num_pts)
                if normalize:
                    for d in range(domain.dim):
                        a, b = domain.intervals[d]
                        data[d] = (data[d] - a) / (b - a)
                combined_points.append(tf.concat(data, axis=1))
            combined_points = tf.concat(combined_points, axis=1)
            for d in range(combined_dim):
                columns['x' + str(d)] = combined_points[:, d].numpy().flatten()
            if j==0:
                pd.DataFrame(columns, columns=list(columns)).to_csv(self.db_path + '/domain.csv',\
                                                                    header=False, index=False)
            else:
                with open(self.db_path + '/domain.csv', 'a', newline='') as csv_file:
                    pd.DataFrame(columns).to_csv(csv_file, header=False, index=False)
                csv_file.close()
            print('Chunk #{} has been written.'.format(j))
        for i, _ in enumerate(self.domains):
            self.domains[i].sample_size = int(num_pts/chunk_size)*chunk_size if chunk_size < num_pts else num_pts

    def build_db(self, num_pts=None, chunk_size=int(1e6), normalize=False, db_type='together'):
        self.db_type = db_type
        db_writer = getattr(self, 'build_db_' + self.db_type)
        db_writer(num_pts=num_pts, chunk_size=chunk_size, normalize=normalize)

    def open_db(self):
        if self.db_type == 'separate':
            self.db = []
            for i in range(len(self.domains)):
                self.db.append(open(self.db_path + '/domain_{}.csv'.format(i)))
        elif self.db_type == 'together':
            self.db = open(self.db_path + '/domain.csv')

    def close_db(self):
        if self.db_type == 'separate':
            for i in range(len(self.domains)):
                try:
                    self.db[i].close()
                except:
                    pass
        elif self.db_type == 'together':
            try:
                self.db.close()
            except:
                pass

    def read_db(self, num_pts, start=None):
        if start is None:
            if self.db_type == 'separate':
                start = [0] * len(self.domains)
            elif self.db_type == 'together':
                start = 0
        dtype = np.float32 if self.dtype == tf.float32 else np.float64
        if self.db_type == 'separate':
            samples = []
            for i, domain in enumerate(self.domains):
                data = np.genfromtxt(self.db[i], dtype=dtype, delimiter=',', skip_header=start[i], max_rows=num_pts[i])
                samples.append(tf.convert_to_tensor(data, dtype=self.dtype))
            return samples
        else:
            data = np.genfromtxt(self.db, dtype=dtype, delimiter=',', skip_header=start, max_rows=num_pts)
            return tf.convert_to_tensor(data, dtype=self.dtype)


class NNSolver(tf.keras.models.Model):
    """
    Description: Class for implementing neural network equation solver
    """
    def __init__(self, name = 'NNSolver', data_path='../../data', dpl_type='h5', dtype=tf.float32):
        super(NNSolver, self).__init__(name=name, dtype=dtype)
        self.folder = data_path + '/{}'.format(self.name)
        self.dpl_type = dpl_type
        try:
            os.mkdir(self.folder)
        except:
            pass
        if dpl_type == 'h5':
            self.db_path = self.folder + '/domains.h5'
            self.dpl = DataPipelineH5(db_path=self.db_path, dtype=self.dtype)
        elif dpl_type == 'csv':
            self.db_path = self.folder
            self.dpl = DataPipelineCSV(db_path=self.db_path, dtype=self.dtype)
        self.domains = NoDependency(self.dpl.domains)
        self.objectives = []

    def add_domain(self, *args):
        """
        Description:
            adds a new doamin
        Args:
            domain: domain to be added
        """
        self.dpl.add_domain(*args)

    def add_objective(self, objective, mean_square=False):
        """
        Description:
            adds a new objective to be minimized
        Args:
            objective: objective to be added
        """
        if mean_square:
            self.objectives.append(self.mean_square(objective))
        else:
            self.objectives.append(objective)

    def mean_square(self, func):
        """
        Description:
            a wrapper for squaring and averaging
        Args:
            func: func to wrap
        Returns:
            wrapped function
        """
        def new_func(*args, **kwargs):
        	val = func(*args,**kwargs)
        	return tf.reduce_mean(tf.square(val))
        return new_func

    def normalize(self, domain_index, *args):
        new_args = []
        for i in range(len(args)):
            a, b = self.dpl.domains[domain_index].intervals[i]
            new_args.append((args[i] - a) / (b - a))
        return new_args

    def build_db(self, num_pts, chunk_size=int(1e5), normalize=True):
        self.dpl.build_db(num_pts=num_pts, chunk_size=chunk_size, normalize=normalize)

    def save_weights(self):
        super().save_weights(self.folder + '\model')

    def load_weights(self):
        super().load_weights(self.folder + '\model').expect_partial()

    def call(self, input):
        # partition inputs
        inputs = tf.split(input, num_or_size_splits=[d.dim for d in self.dpl.domains], axis=1)
        losses = []
        for k, objective in enumerate(self.objectives):
            losses.append(objective(*tf.split(inputs[k], num_or_size_splits=self.dpl.domains[k].dim, axis=1)))
        return tf.reshape(tf.stack(losses, axis=0), (-1, 1))


    def train_step(self, data):
        """
        Description:
            learns the required solution
        Args:
            num_steps: number of steps to train
            num_samples: number of samples to draw from each domain at every step, type = list/array
            initial_rate: intial learning rate
            threshold: stopping threshold for loss
            epochs_per_draw: number of times the solution is to be learned from a batch of samples
        """
        x, y = data
        losses = [0 for i in range(len(self.objectives))]
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        grads = tape.gradient(loss, self.trainable_weights)
        optimizer.apply_gradients(zip(grads, self.trainable_weights))
