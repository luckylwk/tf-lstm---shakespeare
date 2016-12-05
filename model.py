import time
import os

import numpy as np

import tensorflow as tf


# ------------------------------- #
# ------------------------------- #
class Model(object):
    
    
    # ------------------------------- #
    def __init__(self, VOCAB_SIZE, isTraining=False, isSampling=True, 
                 BATCH_SIZE=25, SEQUENCE_LENGTH=50,
                 LAYER_TYPE='lstm', LAYER_SIZE=128, NUM_LAYERS=2):
        
        tf.reset_default_graph()
        
        assert isTraining != isSampling
        
        self.VOCAB_SIZE = VOCAB_SIZE

        self.BATCH_SIZE = BATCH_SIZE
        self.SEQUENCE_LENGTH = SEQUENCE_LENGTH
        self.LEARNING_RATE = 0.05
        self.LEARNING_RATE_DECAY_RATE = 0.97
        self.GRAD_CLIP = 5
        
        self.LAYER_TYPE = LAYER_TYPE
        self.LAYER_SIZE = LAYER_SIZE
        self.NUM_LAYERS = NUM_LAYERS
        
        if isSampling:
            self.BATCH_SIZE = 1
            self.SEQUENCE_LENGTH = 1
        # ------------------------------- #
        
        
    # ------------------------------- #
    def create(self, directory=''):
        
        # Create the recurrent cell(s).
        self.__setupRecurrentCells()
        
        # Create the placeholders
        self.X_data = tf.placeholder(tf.int32, [self.BATCH_SIZE, self.SEQUENCE_LENGTH], name='InputData')
        self.Y_data = tf.placeholder(tf.int32, [self.BATCH_SIZE, self.SEQUENCE_LENGTH], name='OutputLabels')
        
        # Create the embedding parameters
        W = tf.get_variable("softmax_w", [self.LAYER_SIZE, self.VOCAB_SIZE])
        b = tf.get_variable("softmax_b", [self.VOCAB_SIZE])
        embedding = tf.get_variable("embedding", [self.VOCAB_SIZE, self.LAYER_SIZE])
        sth = tf.nn.embedding_lookup(params=embedding, ids=self.X_data, name=None)
        inputs = tf.split(split_dim=1, num_split=self.SEQUENCE_LENGTH, value=sth)
        inputs = [ tf.squeeze(input=input_, squeeze_dims=[1]) for input_ in inputs ]
        #
        outputs, states = tf.nn.seq2seq.rnn_decoder(
            decoder_inputs=inputs, 
            initial_state=self.initial_state, 
            cell=self.cell, 
            loop_function=None
        )
        output = tf.reshape(tf.concat(1, outputs), [-1, self.LAYER_SIZE])
        assert len(outputs) == self.SEQUENCE_LENGTH
        
        self.final_state = states

        # Calculate the probabilities.
        self.logits = tf.nn.bias_add(tf.matmul(output, W), b)
        self.probs = tf.nn.softmax(self.logits)
        
        self.__setupTraining()
        
        self.__setupSession()
        
        # Restore the variables based on the previous checkpoint.
        self.__load(directory=directory)
        
        print 'Model created and setup! Ready to go!\n'
        # ------------------------------- #
    

    # ------------------------------- #
    def __setupRecurrentCells(self):
        # Create the recurrent cell(s).
        if self.LAYER_TYPE == 'lstm':
            cell_ = tf.nn.rnn_cell.BasicLSTMCell(self.LAYER_SIZE, forget_bias=0.0, state_is_tuple=True)
        else:
            cell_ = tf.nn.rnn_cell.BasicRNNCell(self.LAYER_SIZE)
        self.cell = tf.nn.rnn_cell.MultiRNNCell([cell_] * self.NUM_LAYERS, state_is_tuple=True)
        self.initial_state = self.cell.zero_state(self.BATCH_SIZE, tf.float32) 
        print ' --- Created the models recurrent cells'
        # ------------------------------- #


    # ------------------------------- #
    def __setupTraining(self):
    	# Create the loss and cost for the model.
    	self.loss = tf.nn.seq2seq.sequence_loss_by_example(
            logits=[self.logits],
            targets=[tf.reshape(self.Y_data, [-1])],
            weights=[tf.ones([self.BATCH_SIZE * self.SEQUENCE_LENGTH])]
        )
        self.cost = tf.reduce_sum(self.loss)/(self.BATCH_SIZE * self.SEQUENCE_LENGTH)
        # Create a variable for the learning rate. We don't want to train this, but manually set it.
        self.learningRate = tf.Variable(0.0, trainable=False)
        # Get all available trainable variables
        tvars = tf.trainable_variables()
        # We are regularizing by clipping the gradients. 
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), self.GRAD_CLIP)
        # Select the optimizer and the training regime.
        # Maybe switch to RMSProp?
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learningRate)
        self.train_op = self.optimizer.apply_gradients(zip(grads, tvars))
        print ' --- Created the models training operations.'
    	# ------------------------------- #


    # ------------------------------- #
    def __setupSession(self):
    	# Setup the saver module.
    	self.saver = tf.train.Saver(var_list=tf.all_variables())
    	# Create the session and initialise all the variables in the graph.
    	self.sess = tf.Session()
    	self.sess.run(tf.initialize_all_variables())
        print ' --- Initialised the models session and variables.'
    # ------------------------------- #


    # ------------------------------- #
    def __createDataMatrices(self, data, ydata):
        self.num_sequences = len(data) // self.SEQUENCE_LENGTH
        X = np.asarray(data[:self.SEQUENCE_LENGTH*self.num_sequences]).reshape((self.num_sequences,-1))
        Y = np.asarray(ydata[:self.SEQUENCE_LENGTH*self.num_sequences]).reshape((self.num_sequences,-1))
        return X, Y
    # ------------------------------- #

    
    # ------------------------------- #
    def train(self, data, ydata, NUM_EPOCHS=5, start_epoch=0, directory=''):
        
        # Create the datamatrices
        X, Y = self.__createDataMatrices(data, ydata)

        NUM_BATCHES = self.num_sequences // self.BATCH_SIZE

        # Start the training...
        for e in range(start_epoch, NUM_EPOCHS + start_epoch):
            
            # Update the learning rate using the decay.
            self.sess.run(tf.assign(self.learningRate, self.LEARNING_RATE * (self.LEARNING_RATE_DECAY_RATE ** e)))
            
            # Capture the current state.
            current_state = self.sess.run(self.initial_state)
            
            # Create a random permutation of the data. We do this ONLY at the beginning of each EPOCH
            # Permutation is a shuffled list of indices.
            permutation = np.random.permutation(self.num_sequences)
            
            # Start the time...
            t = time.time()
            batch_loss = []
            print 'Starting Epoch {:2d} over {:6d} batches'.format(e+1, NUM_BATCHES)
            print_every_xbatches = int(0.2 * NUM_BATCHES)
            
            for b in range(NUM_BATCHES):
                
                datafeed = {
                    self.X_data: X[permutation[b*self.BATCH_SIZE:(b+1)*self.BATCH_SIZE],:], 
                    self.Y_data: Y[permutation[b*self.BATCH_SIZE:(b+1)*self.BATCH_SIZE],:], 
                    self.initial_state: current_state
                }
                train_loss, current_state, _ = self.sess.run( [ self.cost, self.final_state, self.train_op ], datafeed )
                
                # The newly calculated final state has now become the new current state.
                batch_loss.append(train_loss)
                
                if (b+1) % print_every_xbatches == 0:
                    print '   --- Batch {:6d} completed'.format(b+1)
            
            
            # Print the batch loss
            print '   >>> Epoch {:2d} mean batch loss: {:6.2f} \n'.format(e+1, np.mean(batch_loss))
            
            # Save at the end of training...
            self.__save(directory, e)
            
        print '\nDONE TRAINING!'
        
        # Close the session.
        self.sess.close()
        # ------------------------------- #
    
        
    # ------------------------------- #
    def sample(self, dictionary, reverse_dictionary, seed=['hello', 'my']):
        
        # Start with an empty state.
        current_state = self.sess.run(self.cell.zero_state(self.BATCH_SIZE, tf.float32))
        
        # Update the state based on the seeded words.
        for word in seed:
            x = np.zeros((1, 1))
            x[0, 0] = dictionary[word]
            datafeed = { self.X_data: x, self.initial_state: current_state }
            [current_state] = self.sess.run([ self.final_state ], datafeed)
            
        # Now start sampling predictions
        sampled = seed
        last_word = sampled[-1]
        for n in xrange(300):
            x = np.zeros((1, 1))
            x[0, 0] = dictionary[last_word]
            datafeed = { self.X_data: x, self.initial_state: current_state }
            [prob, state] = self.sess.run([ self.probs, self.final_state ], datafeed)
            sample_ix = self.__fn_WeightedPick(prob[0])
            pred_word = reverse_dictionary[sample_ix]
            # print pred_word
            sampled.append(pred_word)
            last_word = pred_word
        
        print ' '.join(sampled).replace('<START> ','').replace(' <END>','.')
        
        # Close the session.
        self.sess.close()
        # ------------------------------- #
        
        
    # ------------------------------- #
    def __fn_WeightedPick(self, weights):
        t = np.cumsum(weights)
        s = np.sum(weights)
        return (int(np.searchsorted(t, np.random.rand(1)*s)))        
        # ------------------------------- #
        
    
    # ------------------------------- #
    def __save(self, directory='', step=0):
        self.saver.save(sess=self.sess, save_path=os.path.join(directory, 'model-dump.ckpt'), global_step=step)
        # ------------------------------- #
    
    
    # ------------------------------- #
    def __load(self, directory=''):
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=directory)
        if ckpt is not None:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print "Model loaded from {}\n".format(ckpt.model_checkpoint_path)
        # ------------------------------- #

# ------------------------------- #
# ------------------------------- #