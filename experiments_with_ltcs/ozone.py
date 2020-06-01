import numpy as np
import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Run on CPU

import tensorflow as tf
import ltc_model as ltc
from ctrnn_model import CTRNN, NODE, CTGRU
import argparse
import datetime as dt

def to_float(v):
    if(v == "?"):
        return 0
    else:
        return float(v)

def load_trace():
    all_x = []
    all_y = []

    with open("data/ozone/eighthr.data","r") as f:
        miss = 0
        total = 0
        while True:
            line = f.readline()
            if(line is None):
                break
            line = line[:-1]
            parts = line.split(',')

            total+=1
            for i in range(1,len(parts)-1):
                if(parts[i]=="?"):
                    miss+=1
                    break

            if(len(parts)!=74):
                break
            label = int(float(parts[-1]))
            feats = [to_float(parts[i]) for i in range(1,len(parts)-1)]

            all_x.append(np.array(feats))
            all_y.append(label)
    print("Missing features in {} out of {} samples ({:0.2f})".format(miss,total,100*miss/total))
    print("Read {} lines".format(len(all_x)))
    all_x = np.stack(all_x,axis=0)
    all_y = np.array(all_y)

    print("Imbalance: {:0.2f}%".format(100*np.mean(all_y)))
    all_x -= np.mean(all_x) #normalize
    all_x /= np.std(all_x) #normalize

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


class OzoneData:

    def __init__(self,seq_len=32):

        x,y = load_trace()
        
        train_x,train_y = cut_in_sequences(x,y,seq_len,inc=4)

        self.train_x = np.stack(train_x,axis=1)
        self.train_y = np.stack(train_y,axis=1)

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

class OzoneModel:

    def __init__(self,model_type,model_size,sparsity_level=0.0,learning_rate = 0.001):
        self.model_type = model_type
        self.constrain_op = []
        self.sparsity_level = sparsity_level

        self.x = tf.placeholder(dtype=tf.float32,shape=[None,None,72])
        self.target_y = tf.placeholder(dtype=tf.int64,shape=[None,None])

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
            self.constrain_op.extend(self.wm.get_param_constrain_op())
        elif(model_type == "node"):
            self.fused_cell = NODE(model_size,cell_clip=-1)
            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        elif(model_type == "ctgru"):
            self.fused_cell = CTGRU(model_size,cell_clip=-1)
            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        elif(model_type == "ctrnn"):
            self.fused_cell = CTRNN(model_size,cell_clip=-1,global_feedback=True)
            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        else:
            raise ValueError("Unknown model type '{}'".format(model_type))
        
        self._debug_list_sparse_vars = []
        if(self.sparsity_level > 0):
            self.constrain_op.extend(self.get_sparsity_ops())

        self.y = tf.layers.Dense(2,activation=None)(head)
        print("logit shape: ",str(self.y.shape))
        weight = tf.cast(self.target_y,dtype=tf.float32)*1.5+0.1
        self.loss = tf.losses.sparse_softmax_cross_entropy(
            labels = self.target_y,
            logits = self.y,
            weights=weight
            )
        print("loss shape: ",str(self.loss.shape))
        self.loss = tf.reduce_mean(self.loss)

        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_step = optimizer.minimize(self.loss)

        model_prediction = tf.argmax(input=self.y, axis=2)

        lab = tf.cast(self.target_y,dtype=tf.float32)
        pred = tf.cast(model_prediction,dtype=tf.float32)

        # True/False positives/negatives
        tp = tf.reduce_sum(lab*pred)
        tn = tf.reduce_sum((1-lab)*(1-pred))
        fp = tf.reduce_sum((1-lab)*(pred))
        fn = tf.reduce_sum((lab)*(1-pred))

        # don't divide by zero
        # Precision and Recall
        self.prec = tp/(tp+fp+0.00001)
        self.recall = tp/(tp+fn+0.00001)
        # F1-score (Geometric mean of precision and recall)
        self.accuracy = 2*(self.prec*self.recall)/(self.prec+self.recall+0.000001)

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

        self.result_file = os.path.join("results","ozone","{}_{}_{:02d}.csv".format(model_type,model_size,int(100*self.sparsity_level)))
        if(not os.path.exists("results/ozone")):
            os.makedirs("results/ozone")
        if(not os.path.isfile(self.result_file)):
            with open(self.result_file,"w") as f:
                f.write("best epoch, train loss, train acc, valid loss, valid acc, test loss, test acc\n")

        self.checkpoint_path = os.path.join("tf_sessions","ozone","{}".format(model_type))
        if(not os.path.exists("tf_sessions/ozone")):
            os.makedirs("tf_sessions/ozone")
            
        self.saver = tf.train.Saver()

    def get_sparsity_ops(self):
        tf_vars = tf.trainable_variables()
        op_list = []
        self._debug_list_sparse_vars = []
        for v in tf_vars:
            # print("Variable {}".format(str(v)))
            sparsify = False
            if(v.name.startswith("rnn")):
                sparsify = True
                if(len(v.shape)<2):
                    # Don't sparsity biases
                    sparsify = False
                if("ltc" in v.name and (not "W:0" in v.name)):
                    # LTC can be sparsified by only setting w[i,j] to 0
                    # both input and recurrent matrix will be sparsified
                    sparsify = False
            if(sparsify):
                op_list.append(self.sparse_var(v,self.sparsity_level))
            else:
                print("Don't sparsify '{}'".format(v.name))
                
        return op_list
        
    def sparse_var(self,v,sparsity_level):
        mask = np.random.choice([0, 1], size=v.shape, p=[sparsity_level,1-sparsity_level]).astype(np.float32)
        print("Mask mean: ",str(np.mean(mask)))
        v_assign_op = tf.assign(v,v*mask)
        print("Var[{}] will be sparsified with {:0.2f} sparsity level".format(
            v.name,sparsity_level
        ))
        self._debug_list_sparse_vars.append(v)
        return v_assign_op

    def save(self):
        self.saver.save(self.sess, self.checkpoint_path)

    def restore(self):
        self.saver.restore(self.sess, self.checkpoint_path)


    def fit(self,gesture_data,epochs,verbose=True,log_period=50):

        best_valid_acc = 0
        best_valid_stats = (0,0,0,0,0,0,0)
        self.save()
        for e in range(epochs):
            if(verbose and e%log_period == 0):
                test_acc,test_loss = self.sess.run([self.accuracy,self.loss],{self.x:gesture_data.test_x,self.target_y: gesture_data.test_y})
                valid_acc,valid_loss = self.sess.run([self.accuracy,self.loss],{self.x:gesture_data.valid_x,self.target_y: gesture_data.valid_y})
                valid_prec,valid_recall = self.sess.run([self.prec,self.recall],{self.x:gesture_data.valid_x,self.target_y: gesture_data.valid_y})
                print("valid prec: {:0.2f}, recall: {:0.2f}".format(100*valid_prec,100*valid_recall))
                # F1 metric -> higher is better
                if((valid_acc > best_valid_acc and e > 0) or e==1):
                    best_valid_acc = valid_acc
                    best_valid_stats = (
                        e,
                        np.mean(losses),np.mean(accs),
                        valid_loss,valid_acc,
                        test_loss,test_acc
                    )
                    self.save()

            losses = []
            accs = []
            for batch_x,batch_y in gesture_data.iterate_train(batch_size=16):
                acc,loss,_ = self.sess.run([self.accuracy,self.loss,self.train_step],{self.x:batch_x,self.target_y: batch_y})
                if(len(self.constrain_op) > 0):
                    self.sess.run(self.constrain_op)

                losses.append(loss)
                accs.append(acc)

            if(verbose and e%log_period == 0):
                print("Epochs {:03d}, train loss: {:0.2f}, train acc: {:0.2f}, valid loss: {:0.2f}, valid acc: {:0.2f}, test loss: {:0.2f}, test acc: {:0.2f}".format(
                    e,
                    np.mean(losses),np.mean(accs),
                    valid_loss,valid_acc,
                    test_loss,test_acc
                ))
            if(e > 0 and (not np.isfinite(np.mean(losses)))):
                break
        self.restore()
        best_epoch,train_loss,train_acc,valid_loss,valid_acc,test_loss,test_acc = best_valid_stats
        print("Best epoch {:03d}, train loss: {:0.2f}, train acc: {:0.2f}, valid loss: {:0.2f}, valid acc: {:0.2f}, test loss: {:0.2f}, test acc: {:0.2f}".format(
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

    # https://archive.ics.uci.edu/ml/datasets/Ozone+Level+Detection
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default="lstm")
    parser.add_argument('--log',default=1,type=int)
    parser.add_argument('--size',default=32,type=int)
    parser.add_argument('--epochs',default=200,type=int)
    parser.add_argument('--sparsity',default=0.0,type=float)

    args = parser.parse_args()


    ozone_data = OzoneData()
    model = OzoneModel(model_type = args.model,model_size=args.size,sparsity_level=args.sparsity)

    model.fit(ozone_data,epochs=args.epochs,log_period=args.log)

