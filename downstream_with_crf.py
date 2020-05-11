#coding:utf-8
import tensorflow as tf
import numpy as np 
from sklearn.cross_validation import train_test_split
import time
from sklearn.preprocessing import normalize
import os



learning_rate = 0.01
max_samples = 100000
batch_size = 128
display_step = 100
batchid=0


n_input = 131
n_steps = 10
n_hidden = 128
n_classes = 3

bigbatch = n_steps * batch_size

def softmax(x):
    x_exp = np.exp(x)
    #如果是列向量，则axis=0
    x_sum = np.sum(x_exp, axis = 1, keepdims = True)
    s = x_exp / x_sum    
    return s


def labelprocess(label,n_class=n_classes):
    label_length=len(label)
    label_matrix=np.zeros((label_length,n_class))
    for i,j in enumerate(label): 
       label_matrix[i,int(j)]=1
    return label_matrix

def next_batch(batch_size,train_x,train_y):
    global batchid
    if batchid+batch_size*n_steps > len(train_x):
       batchid = 0
    batch_data = (train_x[batchid:min(batchid+batch_size*n_steps, len(train_y)),:])
    batch_labels = (train_y[batchid:min(batchid + batch_size*n_steps, len(train_y))])
    batchid = min(batchid + batch_size*n_steps, len(train_y))
    return batch_data, batch_labels


 # reset graph
tf.reset_default_graph()


os.environ['CUDA_VISIBLE_DEVICES'] = "0"

a = np.loadtxt('result/psg_all.csv')

#a1 = np.loadtxt('matr_all.csv') 
b = np.loadtxt('data/sleep/PSG/label.csv',dtype=np.int32)
#a=np.c_[a,a1]

##wake-sleep
#wake_sleep_label=[]
#for i in b:
#    if i==0:
#        wake_sleep_label.append(i)
#    else:
#        wake_sleep_label.append(1)
#
#b=wake_sleep_label

#wake-REM-NREM
wake_REM_label=[]
for i in b:
    if i==0:
        wake_REM_label.append(i)
    else:
        if i==4:
            wake_REM_label.append(2)
        else:
            wake_REM_label.append(1)
b=wake_REM_label

##wake-REM-NREM 4class
#fourclass_label=[]
#for i in b:
#    if i==0:
#        fourclass_label.append(i)
#    else:
#        if i==4:
#            fourclass_label.append(3)
#        else: 
#            if i==3:
#                fourclass_label.append(2)
#            else:
#                fourclass_label.append(1)
#            
#b=fourclass_label


print(a.shape)


#a = normalize(a, norm='l2')  

train_x=a[0:int(len(b)*0.7),:]
test_x=a[int(len(b)*0.7):,:]
train_y=b[0:int(len(b)*0.7):]
test_y=b[int(len(b)*0.7):]

#train_x,test_x,train_y,test_y = train_test_split(a,b,test_size=0.3)
label_train= labelprocess(train_y)
label_test= labelprocess(test_y)


x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.int32, [None, n_classes])

sequence_lengths = tf.ones([batch_size,],dtype=tf.int32) *(n_steps)
#labels = tf.placeholder(tf.int32, [bigbatch, None])


weights = tf.Variable(tf.random_normal([2 * n_hidden, n_classes]))
biases = tf.Variable(tf.random_normal([n_classes]))


def BiRNN(x, weights, biases):

    lstm_fw_cell = tf.nn.rnn_cell.GRUCell(n_hidden)
    lstm_bw_cell = tf.nn.rnn_cell.GRUCell(n_hidden)
#    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias = 1.0)
#    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias = 1.0)
    
    (output_fw, output_bw),_ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,
                                                    lstm_bw_cell, x, dtype = tf.float32)
    
    context_rep = tf.concat([output_fw, output_bw], axis=-1)
    return context_rep

context = BiRNN(x, weights, biases)
ntime_steps = tf.shape(context)[1]
context_rep_flat = tf.reshape(context, [-1, 2*n_hidden])
pred = tf.matmul(context_rep_flat, weights) + biases


#CRF
labels=tf.argmax(tf.reshape(y, [-1, ntime_steps, n_classes]), 2)
scores = tf.reshape(pred, [-1, ntime_steps, n_classes])
log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(scores,labels, sequence_lengths)
loss = tf.reduce_mean(-log_likelihood)

viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(scores, transition_params, sequence_lengths) 
v=tf.reshape(viterbi_sequence, [-1])
correct_v = tf.equal(v, tf.argmax(y, 1,output_type=tf.int32))
acc_v = tf.reduce_mean(tf.cast(correct_v, tf.float32))

losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=labels)
mask = tf.sequence_mask(sequence_lengths)
losses = tf.boolean_mask(losses, mask)
loss = tf.reduce_mean(losses)




#Bilstm
pred1=tf.nn.softmax(pred, dim=-1)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = pred, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost+losses)

pred_test=tf.argmax(pred, 1)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


init = tf.global_variables_initializer()
saver = tf.compat.v1.train.Saver()
acclist=[]
prelist=np.array([])
prelist_lstm=np.array([])

with tf.Session() as sess:
    sess.run(init)
#    saver.restore(sess,'./model/psg_wake_sleep_bilstm_self.ckpt')
    
    step = 1
    while step * batch_size < max_samples:
        batch_x, batch_y = next_batch(batch_size,train_x,train_y)
        batch_y_1 = labelprocess(batch_y)
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        acc_crf,_,trans_params,viterbi_seq=sess.run([acc_v,optimizer,transition_params,viterbi_sequence], feed_dict = {x: batch_x, y: batch_y_1})
        if step % display_step == 0:
            acc,pred_test_out = sess.run([accuracy,pred_test], feed_dict = {x: batch_x, y: batch_y_1})
            loss = sess.run(cost, feed_dict = {x: batch_x, y: batch_y_1})
            print('acc_crf',acc_crf)
            print("Iter" + str(step * batch_size) + ", Minibatch Loss = " + \
                "{:.6f}".format(loss) + ", Training Accuracy = " + \
                "{:.5f}".format(acc))
        step += 1
#    saver.save(sess, './model/psg_wake_REM_bilstm_self.ckpt')
    print("Optimization Finished!")
    
    step = 1
    print(len(test_y))
    while step  <  len(test_y)/bigbatch:
        batch_x, batch_y = next_batch(batch_size, test_x, test_y)
        batch_y_1 = labelprocess(batch_y)
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        transition_para,acc,viterbi_seq,acc_crf,pred2=sess.run([transition_params,accuracy,viterbi_sequence,acc_v,pred], feed_dict = {x: batch_x, y: batch_y_1})
        v=viterbi_seq.reshape(-1)
        print('acc_crf',acc_crf)
        print("Testing Accuracy:", acc)
        acclist.append(acc)
        prelist=np.r_[prelist,v] if prelist.size else v
        prelist_lstm=np.r_[prelist_lstm,pred2] if prelist_lstm.size else pred2
        step += 1
        
    batch_x=np.r_[test_x[(step-1)*bigbatch:,:],np.zeros((bigbatch-(len(test_y)-(step-1)*bigbatch),n_input))]
    print(batch_x.shape)
    batch_y = np.r_[test_y[(step-1)*bigbatch:],np.zeros((bigbatch-(len(test_y)-(step-1)*bigbatch)))]
    batch_y_1 = labelprocess(batch_y)
    batch_x = batch_x.reshape((batch_size, n_steps, n_input))
    acc,viterbi_seq,acc_crf,pred2=sess.run([accuracy,viterbi_sequence, acc_v, pred], feed_dict = {x: batch_x, y: batch_y_1})
    v=viterbi_seq.reshape(-1)
    print('acc_crf',acc_crf)
    print("Testing Accuracy:", acc)
    acclist.append(acc)
    prelist=np.r_[prelist,v] if prelist.size else v
    prelist_lstm=np.r_[prelist_lstm,pred2] if prelist_lstm.size else pred2
    prelist=prelist[0:len(test_y)]
    prelist_lstm=prelist_lstm[0:len(test_y)]
    
#    np.savetxt('psg_wake_sleep__y_score.csv',prelist)
    np.savetxt('tran.csv',transition_para)
    np.savetxt('1.csv',prelist_lstm)
    np.savetxt('2.csv',label_test)
    
    
a=np.loadtxt('1.csv')
b=np.loadtxt('2.csv')

correct_pred = np.equal(np.argmax(a, 1), np.argmax(b, 1))
accuracy = np.mean(correct_pred)
print('bilstm:',accuracy)

correct_pred = np.equal(np.argmax(a, 1), np.argmax(b, 1))
accuracy_crf = np.mean(correct_pred)
print('crf:',accuracy_crf)


