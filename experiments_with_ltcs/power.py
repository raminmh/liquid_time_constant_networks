import numpy as np
import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Run on CPU

import tensorflow as tf
import ltc_model as ltc
from ctrnn_model import CTRNN, NODE, CTGRU
import argparse
import datetime as dt

def convert_to_floats(feature_col,memory):
    for i in range(len(feature_col)):
        if(feature_col[i]=="?" or feature_col[i] == "\n"):
            feature_col[i] = memory[i]
        else:
            feature_col[i] = float(feature_col[i])
            memory[i] = feature_col[i]
    return feature_col,memory

def load_crappy_formated_csv():

    all_x = []    
    with open("data/power/household_power_consumption.txt","r") as f:
        lineno = -1
        memory = [i for i in range(7)]
        for line in f:
            lineno += 1
            if(lineno == 0):
                continue
            arr = line.split(";")
            if(len(arr)<8):
                continue
            feature_col = arr[2:]
            feature_col,memory = convert_to_floats(feature_col,memory)
            all_x.append(np.array(feature_col,dtype=np.float32))

    all_x = np.stack(all_x,axis=0)
    all_x -= np.mean(all_x,axis=0) #normalize
    all_x /= np.std(all_x,axis=0) #normalize

    all_y = all_x[:,0].reshape([-1,1])
    all_x = all_x[:,1:]


    return all_x,all_y


def cut_in_sequences(x,y,seq_len,inc=1):

    sequences_x = []
    sequences_y = []

    for s in range(0,x.shape[0] - seq_len,inc):
        start = s
        end = start+seq_len
        sequences_x.append(x[start:end])
        sequences_y.append(y[start:end])

    return np.stack(sequences_x,axis=1),np.stack(sequences_y,axis=1)


class PowerData:

    def __init__(self,seq_len=32):

        x,y = load_crappy_formated_csv()
        
        self.train_x,self.train_y = cut_in_sequences(x,y,seq_len,inc=seq_len)

        print("train_x.shape:",str(self.train_x.shape))
        print("train_y.shape:",str(self.train_y.shape))


        total_seqs = self.train_x.shape[1]
        print("Total number of training sequences: {}".format(total_seqs))
        permutation = np.random.RandomState(23489).permutation(total_seqs)
        valid_size = int(0.1*total_seqs)
        test_size = int(0.15*total_seqs)

        self.valid_x = self.train_x[:,permutation[:valid_size]]
        self.valid_y = self.train_y[:,permutation[:valid_size]]
        self.test_x = self.train_x[:,permutation[valid_size:valid_size+test_size]]
        self.test_y = self.train_y[:,permutation[valid_size:valid_size+test_size]]
        self.train_x = self.train_x[:,permutation[valid_size+test_size:]]
        self.train_y = self.train_y[:,permutation[valid_size+test_size:]]

    def iterate_train(self,batch_size=16):
        total_seqs = self.train_x.shape[1]
        permutation = np.random.permutation(total_seqs)
        total_batches = total_seqs // batch_size

        for i in range(total_batches):
            start = i*batch_size
            end = start + batch_size
            batch_x = self.train_x[:,permutation[start:end]]
            batch_y = self.train_y[:,permutation[start:end]]
            yield (batch_x,batch_y)

class PowerModel:

    def __init__(self,model_type,model_size,learning_rate = 0.001):
        self.model_type = model_type
        self.constrain_op = None
        self.x = tf.placeholder(dtype=tf.float32,shape=[None,None,6])
        self.target_y = tf.placeholder(dtype=tf.float32,shape=[None,None,1])

        self.model_size = model_size
        head = self.x
        if(model_type == "lstm"):
            self.fused_cell = tf.nn.rnn_cell.LSTMCell(model_size)

            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        elif(model_type.startswith("ltc")):
            learning_rate = 0.01 # LTC needs a higher learning rate
            self.wm = ltc.LTCCell(model_size)
            if(model_type.endswith("_rk")):
                self.wm._solver = ltc.ODESolver.RungeKutta
            elif(model_type.endswith("_ex")):
                self.wm._solver = ltc.ODESolver.Explicit
            else:
                self.wm._solver = ltc.ODESolver.SemiImplicit

            head,_ = tf.nn.dynamic_rnn(self.wm,head,dtype=tf.float32,time_major=True)
            self.constrain_op = self.wm.get_param_constrain_op()
        elif(model_type == "node"):
            self.fused_cell = NODE(model_size,cell_clip=10)
            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        elif(model_type == "ctgru"):
            self.fused_cell = CTGRU(model_size,cell_clip=-1)
            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        elif(model_type == "ctrnn"):
            self.fused_cell = CTRNN(model_size,cell_clip=-1,global_feedback=True)
            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        else:
            raise ValueError("Unknown model type '{}'".format(model_type))

        # target_y = tf.expand_dims(self.target_y,axis=-1)
        self.y = tf.layers.Dense(1,activation=None,kernel_initializer=tf.keras.initializers.TruncatedNormal())(head)
        print("logit shape: ",str(self.y.shape))
        self.loss = tf.reduce_mean(tf.square(self.target_y-self.y))
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_step = optimizer.minimize(self.loss)

        self.accuracy = tf.reduce_mean(tf.abs(self.target_y-self.y))

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

        self.result_file = os.path.join("results","power","{}_{}.csv".format(model_type,model_size))
        if(not os.path.exists("results/power")):
            os.makedirs("results/power")
        if(not os.path.isfile(self.result_file)):
            with open(self.result_file,"w") as f:
                f.write("best epoch, train loss, train mae, valid loss, valid mae, test loss, test mae\n")

        self.checkpoint_path = os.path.join("tf_sessions","power","{}".format(model_type))
        if(not os.path.exists("tf_sessions/power")):
            os.makedirs("tf_sessions/power")
            
        self.saver = tf.train.Saver()

    def save(self):
        self.saver.save(self.sess, self.checkpoint_path)

    def restore(self):
        self.saver.restore(self.sess, self.checkpoint_path)


    def fit(self,power_data,epochs,verbose=True,log_period=50):

        best_valid_loss = np.PINF
        best_valid_stats = (0,0,0,0,0,0,0)
        self.save()
        for e in range(epochs):
            if(verbose and e%log_period == 0):
                print("self.target_y:    ",str(self.target_y.shape))
                print("power_data.test_y ",str(power_data.test_y.shape))
                test_acc,test_loss = self.sess.run([self.accuracy,self.loss],{self.x:power_data.test_x,self.target_y: power_data.test_y})
                valid_acc,valid_loss = self.sess.run([self.accuracy,self.loss],{self.x:power_data.valid_x,self.target_y: power_data.valid_y})
                # MSE metric -> less is better
                if((valid_loss < best_valid_loss and e > 0) or e==1):
                    best_valid_loss = valid_loss
                    best_valid_stats = (
                        e,
                        np.mean(losses),np.mean(accs),
                        valid_loss,valid_acc,
                        test_loss,test_acc
                    )
                    self.save()

            losses = []
            accs = []
            for batch_x,batch_y in power_data.iterate_train(batch_size=16):
                acc,loss,_ = self.sess.run([self.accuracy,self.loss,self.train_step],{self.x:batch_x,self.target_y: batch_y})
                if(not self.constrain_op is None):
                    self.sess.run(self.constrain_op)

                losses.append(loss)
                accs.append(acc)

            if(verbose and e%log_period == 0):
                print("Epochs {:03d}, train loss: {:0.2f}, train mae: {:0.2f}, valid loss: {:0.2f}, valid mae: {:0.2f}, test loss: {:0.2f}, test mae: {:0.2f}".format(
                    e,
                    np.mean(losses),np.mean(accs),
                    valid_loss,valid_acc,
                    test_loss,test_acc
                ))
            if(e > 0 and (not np.isfinite(np.mean(losses)))):
                break
        self.restore()
        best_epoch,train_loss,train_acc,valid_loss,valid_acc,test_loss,test_acc = best_valid_stats
        print("Best epoch {:03d}, train loss: {:0.2f}, train mae: {:0.2f}, valid loss: {:0.2f}, valid mae: {:0.2f}, test loss: {:0.2f}, test mae: {:0.2f}".format(
            best_epoch,
            train_loss,train_acc,
            valid_loss,valid_acc,
            test_loss,test_acc
        ))
        with open(self.result_file,"a") as f:
            f.write("{:08d}, {:0.8f}, {:0.8f}, {:0.8f}, {:0.8f}, {:0.8f}, {:0.8f}\n".format(
            best_epoch,
            train_loss,train_acc,
            valid_loss,valid_acc,
            test_loss,test_acc
        ))

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default="lstm")
    parser.add_argument('--log',default=1,type=int)
    parser.add_argument('--size',default=32,type=int)
    parser.add_argument('--epochs',default=200,type=int)
    args = parser.parse_args()


    power_data = PowerData()
    model = PowerModel(model_type = args.model,model_size=args.size)

    model.fit(power_data,epochs=args.epochs,log_period=args.log)

