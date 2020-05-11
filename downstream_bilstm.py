#coding:utf-8
import tensorflow as tf
import numpy as np 
from sklearn.cross_validation import train_test_split
import time
from sklearn.preprocessing import normalize
import os



learning_rate = 0.001
max_samples = 700000
batch_size = 128
display_step = 100
batchid=0


n_input = 2
n_steps = 2
n_hidden = 128
n_classes = 4


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
    if batchid+batch_size > len(train_x):
       batchid = 0
    batch_data = (train_x[batchid:min(batchid+batch_size, len(train_y)),:])
    batch_labels = (train_y[batchid:min(batchid + batch_size, len(train_y))])
    batchid = min(batchid + batch_size, len(train_y))
    return batch_data, batch_labels


 # reset graph
tf.reset_default_graph()


os.environ['CUDA_VISIBLE_DEVICES'] = "0"

a = np.loadtxt('data/sleep/PSG/data.csv')
#a1 = np.loadtxt('matr_all.csv') 
b = np.loadtxt('data/sleep/PSG/label.csv',dtype=np.int32)

#a=np.c_[a,a1]
#b=b-1
##wake-sleep
#wake_sleep_label=[]
#for i in b:
#    if i==0:
#        wake_sleep_label.append(i)
#    else:
#        wake_sleep_label.append(1)
#
#b=wake_sleep_label

##wake-REM-NREM
#wake_REM_label=[]
#for i in b:
#    if i==0:
#        wake_REM_label.append(i)
#    else:
#        if i==4:
#            wake_REM_label.append(2)
#        else:
#            wake_REM_label.append(1)
#b=wake_REM_label

#wake-REM-NREM 4class
fourclass_label=[]
for i in b:
    if i==0:
        fourclass_label.append(i)
    else:
        if i==4:
            fourclass_label.append(3)
        else: 
            if i==3:
                fourclass_label.append(2)
            else:
                fourclass_label.append(1)
            
b=fourclass_label

print(a.shape)
#a = normalize(a, norm='l2')  

#train_x=a[0:int(len(b)*0.8),:]
#test_x=a[int(len(b)*0.8):,:]
#train_y=b[0:int(len(b)*0.8):]+1
#test_y=b[int(len(b)*0.8):]+1

train_x,test_x,train_y,test_y = train_test_split(a,b,test_size=0.3)
label_train= labelprocess(train_y)
label_test= labelprocess(test_y)


x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

weights = tf.Variable(tf.random_normal([2 * n_hidden, n_classes]))
biases = tf.Variable(tf.random_normal([n_classes]))


def BiRNN(x, weights, biases):

    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(x, n_steps)

    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias = 1.0)
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias = 1.0)
    outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell,
                                                            lstm_bw_cell, x,
                                                            dtype = tf.float32)
    return tf.matmul(outputs[-1], weights) + biases, outputs[-1]

pred,out = BiRNN(x, weights, biases)
print(pred,out)

pred1=tf.nn.softmax(pred, dim=-1, name=None)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = pred, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)


pred_test=tf.argmax(pred, 1)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
saver = tf.compat.v1.train.Saver()


acclist=[]
losslist=[]

tf.logging.info('Graph loaded')

acclists_test=[]

with tf.Session() as sess:
    sess.run(init)
    step = 1
#    saver.restore(sess,'./model/psg_wake_REM_bilstm_self.ckpt')    
    while step * batch_size < max_samples:
        batch_x, batch_y = next_batch(batch_size,train_x,train_y)
        batch_y_1 = labelprocess(batch_y)
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        sess.run(optimizer, feed_dict = {x: batch_x, y: batch_y_1})
        
        if step % display_step == 0:
            acc,pred_test_out = sess.run([accuracy,pred_test], feed_dict = {x: batch_x, y: batch_y_1})
            acclist.append(acc)
            loss = sess.run(cost, feed_dict = {x: batch_x, y: batch_y_1})
            losslist.append(loss)
            print("Iter" + str(step * batch_size) + ", Minibatch Loss = " + \
                "{:.6f}".format(loss) + ", Training Accuracy = " + \
                "{:.5f}".format(acc))
            
        step += 1
    print("Optimization Finished!")
#
#    saver.save(sess, './model/psg_wake_REM_bilstm_self.ckpt')
    tf.logging.info('Training done')
#    

    test_data = test_x.reshape((-1,n_steps, n_input))
    test_label = label_test
    pred_score,outfea=sess.run([pred,out], feed_dict = {x: test_data, y: test_label})
    print("Testing Accuracy:", sess.run(accuracy, feed_dict = {x: test_data, y: test_label}))
    np.savetxt('psg_fourclass_y_score.csv',pred_score)
    np.savetxt('psg_fourclass_test_score.csv',test_label)
    
    
#a=np.loadtxt('psg_wake_REM_y_score.csv')
#b=np.loadtxt('psg_wake_REM_test_score.csv')
#correct_pred = np.equal(np.argmax(a, 1), np.argmax(b, 1))
#accuracy = np.mean(correct_pred)
#print(accuracy)


