'''
Stanford CS233, HW4.

Written by Panos Achlioptas, 2018.
'''


import tensorflow as tf
import numpy as np
import time
import os.path as osp
from six.moves import xrange

from nn_distance_cpu import nn_distance_cpu as nn_distance
from neural_net import Neural_Net, Neural_Net_Conf, MODEL_SAVER_ID

from tf_util import conv1d as conv_1d
relu = tf.nn.relu


class cs233PointAutoEncoder(Neural_Net):
    ''' A versatile AE on point-clouds, that allows for simultaneous learning on part-segmentations.
    
    Students: do filling.
    '''
    def __init__(self, name, configuration, graph=None):
        Neural_Net.__init__(self, name, graph)
        self.config = configuration
        c = configuration
        
        self.in_pc = tf.placeholder(tf.float32, shape = (50, 1024, 3), name = 'input_pcs')      # Students. Create appropriate placeholders.
            
        if c.use_parts:
            self.part_mask = tf.placeholder(tf.int32, shape = (50, 1024), name = 'input_part_masks')
                    
        with tf.variable_scope(name):            
            self.z = c.encoder(self.in_pc, **c.encoder_args)            # Encoder registration.
            dec_out = c.decoder(self.z, **c.decoder_args)               # Decoder registration.
            self.bottleneck_size = int(self.z.get_shape()[1])
            
            self.pc_reconstr = tf.reshape(dec_out, self.in_pc.shape, name = 'reconstructed_pcs') # Students. reshape?
                                    
            if c.use_parts:                    
                if c.part_pred_with_one_layer:                                    
                    self.part_pred = 0 # Dan & Liz: You should change this
                else:                    
                    self.part_pred = 0 # Dan & Liz: You should change this
                                        
        # Add Reconstruction Loss.
        dist1, idx1, dist2, idx2 = nn_distance(self.in_pc, self.pc_reconstr)
        self.recon_loss = tf.reduce_mean(dist1 + dist2)           
        
        self.part_loss = 0
        if c.use_parts:
            # Add X-entropy Loss.
            pass # Dan & Liz: You should change this
        # Dan & Liz: See the Piazza post "HW 4. Defining and combining the loss." for explanation
        self.total_loss = self.recon_loss + c.part_weight*self.part_loss # Students.

        # Optimizer
        self.lr = c.learning_rate
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_step = self.optimizer.minimize(self.total_loss)
        
        # Auxiliary.
        self.no_op = tf.no_op()
        if c.use_parts:
            # Define Part Prediction (accuracy) ops.
            pass # Dan & Liz: You should change this
        
        
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=c.saver_max_to_keep)         
        self.start_session(allow_growth=c.allow_gpu_growth)
                
    def prepare_feed(self, dataset, batch_size):
        '''
        Extract a mini-batch of (batch_size) examples from the input dataset. 
        Checks the AE's configuration to decide to fetch PCs or PCs and Part-Masks.
        '''
        if self.config.use_parts:            
            pc_b, mask_b = dataset.next_batch(batch_size, ['pcs', 'part_masks'])
            feed_dict = {self.in_pc: pc_b, self.part_mask: mask_b}
        else:
            pc_b = dataset.next_batch(batch_size, ['pcs'])[0]            
            feed_dict = {self.in_pc: pc_b}
        return feed_dict
            
    def _single_epoch_train(self, train_data, batch_size, only_fw=False):
        '''
            Trains the AE for a single epoch of the provided (NumpyDataset) train_data.

            If only_fw is True, back-prop in not computed/applied (this is exploited e.g. by the validation dataset).
        '''
        n_examples = train_data.n_examples
        n_batches = int(n_examples / batch_size)
            
        if only_fw:
            fit = self.no_op
        else:
            fit = self.train_step
            
        if self.config.use_parts:
            # You need to inspect two losses.
            epoch_loss = np.array([0.0, 0.0])
            compute_operations = [fit, self.recon_loss, self.part_loss]
        else:
            epoch_loss = np.array([0.0])
            compute_operations = [fit, self.recon_loss]
            
        start_time = time.time()         
        
        # Students. Loop over all batches to process the compute_operations and gather loss.
        
        for _ in xrange(n_batches):
                feed = self.prepare_feed(train_data, batch_size)  
                epoch_loss += self.sess.run(compute_operations, feed_dict=feed)[1:]
        
        duration = time.time() - start_time
        epoch_loss = epoch_loss/n_batches

        return epoch_loss, duration
    
    def train_model(self, net_data, n_epochs, batch_size, save_dir, held_out_step=10,
                    verbose=True, fout=None):
        '''
        Trainer of the AE.
        Args:
            net_data: an appropriate (NumpyDataset) dataset with point-clouds and optionally part-masks.
            n_epochs: how many epochs to train the AE.
            batch_size: number of examples in each mini-batch
            save_dir: where to save the AE's learned weights 
            held_out_step: (int) every this epochs the system will evaluate the losses on the test/val 
            data and will store the current weights of the AE if the validation loss improved.
            fout: a (File) to write the training statistics.
        '''
        
        train_loss = []
        val_loss = []
        test_loss = []
        val_loss_best = np.Inf
        checkpoint_path = osp.join(save_dir, MODEL_SAVER_ID)
        
        for _ in range(1, n_epochs + 1):
            tr_loss, dur = self._single_epoch_train(net_data['train'], batch_size)
            train_loss.append(tr_loss)
            epoch = int(self.sess.run(self.increment_epoch))

            if verbose:
                print('Training epoch/loss/duration: ', epoch, tr_loss, dur)
            
            if fout is not None:
                fout.write('Training epoch/loss: %d %s\n' % (epoch, str(tr_loss)) )                                
            if epoch % held_out_step == 0:
                v_loss, _ = self._single_epoch_train(net_data['val'], batch_size, only_fw=True)
                val_loss.append(v_loss)
                t_loss, _ = self._single_epoch_train(net_data['test'], batch_size, only_fw=True)
                test_loss.append(t_loss)
                if verbose:
                    print('Val/Test epoch/loss:', epoch, v_loss, t_loss)
                
                if fout is not None:
                    fout.write('Val/Test epoch/loss: %d %s %s\n' % (epoch, str(v_loss), str(t_loss)))                              
                
                # Students: add code that checks if the validation got better. If it did, save the model.
                if v_loss < val_loss_best:
                    val_loss_best = v_loss
                    save_path = osp.join(checkpoint_path, '-'+str(int(_))
                    self.saver.save(self.sess, save_path)
                    

        return np.array(train_loss), np.array(val_loss), np.array(test_loss)
