#coding:utf-8
'''
Aite Zhao

You can change workers and pretraining model by yourselves.

MLP is the most basic training model.
'''
import tensorflow as tf
import numpy as np 
from sklearn.cross_validation import train_test_split
import time
from sklearn.preprocessing import normalize
import tensorflow.contrib.slim as slim
import os
from fishervector import FisherVectorGMM
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import metrics




def FFT(a):
    transy=np.fft.fft(a)  
    return transy

def test_with_gaussian_samples_image(data,n_kernels):
    test_data = np.array(data)
    test_data = test_data.reshape([test_data.shape[0],-1,1])
    print(test_data.shape)
    fv_gmm = FisherVectorGMM(n_kernels=n_kernels).fit(test_data)
    n_test_videos = len(data)
    fv = fv_gmm.predict(test_data[:n_test_videos])
    print(fv.shape)
    return fv

def workers_batch(input_images,input_labels):
    batch_x = input_images[batch_start:batch_end,:]
    batch_y = input_labels[batch_start:batch_end,:]
#    batch_x = batch_x.reshape((batch_size, n_steps, n_input))
    return batch_x, batch_y


def softmax(x):
    x_exp = np.exp(x)
    #如果是列向量，则axis=0
    x_sum = np.sum(x_exp, axis = 1, keepdims = True)
    s = x_exp / x_sum    
    return s


def next_batch(batch_size,x,y):
    global batchid,batch_start,batch_end
    batch_start = batchid
    batch_end = min(batchid + batch_size, m)
    if batchid+batch_size > m:
       batchid = 0
       batch_start=0
       batch_end=min(batchid + batch_size, m)
    batch_data = (x[batchid:min(batchid + batch_size, m),:])
    batch_labels = (y[batchid:min(batchid + batch_size, m),:])
    batchid = min(batchid + batch_size, m)
    return batch_data, batch_labels

learning_rate = 0.001
max_samples = 200000
batch_size = 128
display_step = 100
n_classes = 5
n_worker=5


#bilstm
batchid=0
batch_start=0
batch_end=0
n_input = 64
n_steps = 27
#n_hidden = 128
loss_weight=0.2

def labelprocess(label,n_class=n_classes):
    label_length=len(label)
    label_matrix=np.zeros((label_length,n_class))
    for i,j in enumerate(label): 
       label_matrix[i,int(j)]=1
    return label_matrix


 # reset graph
tf.reset_default_graph()


os.environ['CUDA_VISIBLE_DEVICES'] = "0"

a = np.loadtxt('data/sleep/pressure/matr_all.csv')
a = normalize(a, norm='l2')
m,n=a.shape

a_fisher=np.array([])
#a_fisher=test_with_gaussian_samples_image(a,5)
#np.savetxt('result/matr_fisher.csv',a_fisher.reshape(m,-1)) 

a_pca=np.array([])
#pca = PCA(n_components=3)   
#pca.fit(a)                  
#a_pca=pca.fit_transform(a)
#np.savetxt('result/matr_pca.csv',a_pca)  

rot90_images = np.array([])
rot180_images = np.array([])
rot270_images = np.array([])
fft_images = np.array([])

for i in a:
    r90=np.rot90(i.reshape(n_input,n_steps), 1).reshape(-1)
    r180=np.rot90(i.reshape(n_input,n_steps), 2).reshape(-1)
    r270=np.rot90(i.reshape(n_input,n_steps), 3).reshape(-1)
    fft_fea=FFT(i.real)
    
    rot90_images=np.c_[rot90_images,r90] if rot90_images.size>0 else r90
    rot180_images=np.c_[rot180_images,r180] if rot180_images.size>0 else r180
    rot270_images=np.c_[rot270_images,r270] if rot270_images.size>0 else r270
    fft_images=np.c_[fft_images,fft_fea] if fft_images.size>0 else fft_fea

labels = labelprocess(np.zeros(m))
label90 = labelprocess(np.ones(m)+1)
label180 = labelprocess(np.ones(m)+2)
label270 =  labelprocess(np.ones(m)+3)
label_fft =  labelprocess(np.zeros(m)+4)

input_images = np.c_[np.transpose(a),rot90_images,rot180_images,rot270_images,fft_images]
#input_images = np.c_[np.transpose(a),fft_images]
input_images = np.transpose(input_images)
print(input_images.shape)
input_labels =  np.r_[labels, label90, label180, label270,label_fft]
#input_labels =  np.r_[labels,label_fft]



in_units = 1728
h1_units = 128 
out1 = tf.placeholder(tf.float32, [None, n_classes])
x = tf.placeholder(tf.float32, [None, in_units]) 
y_ = tf.placeholder(tf.float32, [None, n_classes]) 
keep_prob = tf.placeholder(tf.float32) 

def MLP():  
    #h2_units = 128
    W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1)) #初始化隐含层权重W1，服从默认均值为0，标准差为0.1的截断正态分布 
    b1 = tf.Variable(tf.zeros([h1_units])) #隐含层偏置b1全部初始化为0 
    #W2 = tf.Variable(tf.truncated_normal([h1_units, h2_units], stddev=0.1)) #初始化隐含层权重W1，服从默认均值为0，标准差为0.1的截断正态分布 
    #b2 = tf.Variable(tf.zeros([h2_units])) #隐含层偏置b1全部初始化为0 
    W3 = tf.Variable(tf.zeros([h1_units, n_classes]))  
    b3 = tf.Variable(tf.zeros([n_classes])) 
    #定义模型结构 
    hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1) 
    hidden1_drop = tf.nn.dropout(hidden1, keep_prob) 
    #hidden2 =  tf.nn.relu(tf.matmul(hidden1, W2) + b2) 
    #hidden2_drop = tf.nn.dropout(hidden2, keep_prob) 
    y = tf.nn.softmax(tf.matmul(hidden1_drop, W3) + b3) 
    return hidden1,y

#训练部分 
h,y =MLP()
kl_loss=tf.contrib.distributions.kl_divergence(tf.distributions.Categorical(probs=out1),tf.distributions.Categorical(probs=y),allow_nan_stats=False)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = y, labels = y_))
L=tf.reduce_mean(kl_loss)*loss_weight+(1-loss_weight)*cross_entropy
train_step = tf.train.AdamOptimizer(learning_rate).minimize(L) 
sess = tf.InteractiveSession() 
tf.global_variables_initializer().run() 
correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1)) 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 



for i in range(5001): 
  batch_xs, batch_ys = next_batch(batch_size,input_images[0:m,:],input_labels[0:m,:])
  out_f=y.eval({x: batch_xs, y_: batch_ys, keep_prob: 1}) 
  train_step.run({x: batch_xs, y_: batch_ys, out1: out_f, keep_prob: 1}) 
  for j in range(n_classes-1):
    batch_x1,batch_y1 = workers_batch(input_images[(j+1)*m:(j+1)*m+m,:],input_labels[(j+1)*m:(j+1)*m+m,:])
    kl_loss.eval({x: batch_x1, y_: batch_y1, out1: out_f, keep_prob: 1})
  if i % 200 ==0: 
    print(i, 'Loss:', L.eval({x: batch_xs, y_:batch_ys, out1: out_f,
               keep_prob: 0.75}))
    print(i, 'training_arruracy:', accuracy.eval({x: batch_xs, y_:batch_ys, out1: out_f,
               keep_prob: 0.75})) 
    
    
    
    
    
test_data = input_images
test_label = input_labels
print('final_accuracy:', accuracy.eval({x:test_data, y_: test_label, keep_prob: 1.0})) 
outfea = h.eval({x:test_data, y_: test_label, keep_prob: 1.0})
print(outfea.shape,m)
#outfea = np.c_[outfea[0:m,:], outfea[m:2*m,:]]
#outfea = np.c_[outfea[0:m,:], outfea[m:2*m,:], outfea[2*m:3*m,:], outfea[3*m:4*m,:],outfea[4*m:5*m,:]]
#outfea = np.c_[outfea[0:m,:], outfea[m:2*m,:], outfea[2*m:3*m,:]]
np.savetxt('result/ma_mlp.csv',outfea) 
#outfea = np.c_[outfea,a_fisher.reshape((m,-1)),a_pca.reshape((m,-1))]
#np.savetxt('result/ma_all.csv',outfea)   





#x = tf.placeholder("float", [None, n_steps, n_input])
#y = tf.placeholder("float", [None, n_classes])
#
#out1 = tf.placeholder("float", [None, 2 * n_hidden])
#
#weights = tf.Variable(tf.random_normal([2 * n_hidden, n_classes]))
#biases = tf.Variable(tf.random_normal([n_classes]))
#
#
#def workers_batch(input_images,input_labels):
#    batch_x = input_images[batch_start:batch_end,:]
#    batch_y = input_labels[batch_start:batch_end,:]
#    batch_x = batch_x.reshape((batch_size, n_steps, n_input))
#    return batch_x, batch_y
#    
#
#def get_cos_distance(i,j):
#    # calculate cos distance between two sets
#    # more similar more big
#    x3_norm = tf.sqrt(tf.reduce_sum(tf.square(i), axis=0))
#    x4_norm = tf.sqrt(tf.reduce_sum(tf.square(j), axis=0))
#    #内积
#    x3_x4 = tf.reduce_sum(tf.multiply(x3_norm,x4_norm), axis=0)
##    cosin = x3_x4 / (x3_norm * x4_norm)
#    cosin1 = tf.divide(x3_x4, tf.multiply(x3_norm, x4_norm))
##    cosin1 = tf.nn.softmax(cosin1, dim=-1, name=None)
#    return tf.reduce_sum(cosin1)
#
#
#def BiRNN(x, weights, biases):
#
#    x = tf.transpose(x, [1, 0, 2])
#    x = tf.reshape(x, [-1, n_input])
#    x = tf.split(x, n_steps)
#
#    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias = 1.0)
#    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias = 1.0)
#    outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell,
#                                                            lstm_bw_cell, x,
#                                                            dtype = tf.float32)
#    return tf.matmul(outputs[-1], weights) + biases, outputs[-1]
#
#pred,out = BiRNN(x, weights, biases)
#print(pred,out)
#
#
#
#pred1=tf.nn.softmax(pred, dim=-1, name=None)
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = pred, labels = y))
#optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
#
#
##loss_1 = noise_contrastive_estimator(x, out)
##loss_2 = noise_contrastive_estimator(input_images, output[0]) 
#
##cos_loss = get_cos_distance(out, out1)
##cos_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = out1, labels = out))
##optimizer1 = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cos_loss)
#
#
#pred_test=tf.argmax(pred, 1)
#correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#
#init = tf.global_variables_initializer()
#saver = tf.compat.v1.train.Saver()
#
#
#acclist=[]
#losslist=[]
#
#tf.logging.info('Graph loaded')
#
#with tf.Session() as sess:
#    sess.run(init)
#    step = 1
#    
#    while step * batch_size < max_samples:
#        batch_x, batch_y = next_batch(batch_size,input_images,input_labels)
##        batch_x, batch_y = next_batch(batch_size,input_images[0:m,:],input_labels[0:m,:])
#        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
#        _,out_f = sess.run([optimizer,out], feed_dict = {x: batch_x, y: batch_y})
##        print(step,m,batch_start,batch_end)
#        
##        for i in range(n_classes-1):
##            batch_x1,batch_y1 = workers_batch(input_images[(i+1)*m:(i+1)*m+m,:],input_labels[(i+1)*m:(i+1)*m+m,:])
##            sess.run([optimizer,optimizer1], feed_dict = {x: batch_x1, y: batch_y1, out1: out_f})
#        
#        if step % display_step == 0:
#            acc,pred_test_out = sess.run([accuracy,pred_test], feed_dict = {x: batch_x, y: batch_y})
#            acclist.append(acc)
#            loss = sess.run(cost, feed_dict = {x: batch_x, y: batch_y})
##            loss = sess.run(cos_loss, feed_dict = {x: batch_x, y: batch_y, out1: out_f})
#            losslist.append(loss)
#            print("Iter" + str(step * batch_size) + ", Minibatch Loss = " + \
#                "{:.6f}".format(loss) + ", Training Accuracy = " + \
#                "{:.5f}".format(acc))
#            
#        step += 1
#    print("Optimization Finished!")
#
#    saver.save(sess, './model/bio_self.ckpt')
#    tf.logging.info('Training done')
#    
#
#    test_data = input_images.reshape((-1,n_steps, n_input))
#    test_label = input_labels
#    s=time.time()
#    pred_test_out,pred_score,outfea=sess.run([pred_test,pred1,out], feed_dict = {x: test_data, y: test_label})
#    outfea = np.c_[outfea[0:m,:], outfea[m:2*m,:]]
#    outfea = np.c_[outfea[0:m,:], outfea[m:2*m,:], outfea[2*m:3*m,:], outfea[3*m:4*m,:],outfea[4*m:5*m,:]]
#    np.savetxt('result/bio_lstm.csv',outfea) 
#    outfea = np.c_[outfea,a_fisher.reshape((m,-1)),a_pca.reshape((m,-1))]
#    np.savetxt('result/bio_all.csv',outfea)   
#    e=time.time()
#    print("Testing Accuracy:", sess.run(accuracy, feed_dict = {x: test_data, y: test_label}))
#    print(e-s)
