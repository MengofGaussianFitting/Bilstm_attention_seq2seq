# coding: utf-8




import numpy as np
import time
import tensorflow as tf
from tensorflow.python.layers.core import Dense
import json
from tensorflow.python.client import timeline
import pickle
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"








with open('data/source.txt','r')as f:
    source_data = f.read().split('\n')
with open('data/target.txt','r')as f:
    target_data = f.read().split('\n')


valid_source = []
valid_target = []




#source_data = source_data[:381000]# + source_data[390000:]
#target_data = target_data[:381000]# + target_data[390000:]
#source_f = open('datastru/source_data.pkl','r')
#source_data = pickle.load(source_f)
#source_f.close()
#target_f = open('datastru/target_data.pkl','r')
#target_data = pickle.load(target_f)
#target_f.close()

print(len(source_data))
print(len(target_data))


def extract_character_vocab(data):
    special_words = ['<PAD>', '<UNK>', '<GO>', '<EOS>']
    set_words = list(set([character for line in data for character in line.split()]))
    int_to_vocab = {idx:word for idx,word in enumerate(special_words + set_words)}
    vocab_to_int = {word:idx for idx,word in int_to_vocab.items()}
    
    return int_to_vocab,vocab_to_int





source_int_to_letter, source_letter_to_int = extract_character_vocab(source_data)
target_int_to_letter, target_letter_to_int = extract_character_vocab(target_data)
print(sys.getsizeof(source_int_to_letter))

source_int = [[source_letter_to_int.get(letter, source_letter_to_int.get('<UNK>')) for letter in line.split()] for line in source_data]
target_int = [[target_letter_to_int.get(letter, target_letter_to_int.get('<UNK>')) for letter in line.split()] + [target_letter_to_int.get('<EOS>')] for line in target_data]

#sil_f = open('datastru/sil_f.pkl','r')
#source_int_to_letter = pickle.load(sil_f)
#sil_f.close()
#sli_f = open('datastru/sli_f.pkl','r')
#source_letter_to_int = pickle.load(sli_f)
#sli_f.close()
#til_f = open('datastru/til_f.pkl','r')
#target_int_to_letter = pickle.load(til_f)
#til_f.close()
#tli_f = open('datastru/tli_f.pkl','r')
#target_letter_to_int = pickle.load(tli_f)
#tli_f.close()
#si_f = open('datastru/si_f.pkl','r')
#source_int = pickle.load(si_f)
#si_f.close()
#ti_f = open('datastru/ti_f.pkl','r')
#target_int = pickle.load(ti_f)
#ti_f.close()
#print(source_int_to_letter)
#print(target_int_to_letter)



def get_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
  
    target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
    max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_sequence_lgenght')
    source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')
    
    return inputs, targets, learning_rate, target_sequence_length, max_target_sequence_length, source_sequence_length



# Encoder
def get_encoder_layer(input_data, rnn_size, num_layers, 
                      source_sequence_length, source_vocab_size, 
                      encoding_embedding_size):
    encoder_embed_input = tf.contrib.layers.embed_sequence(input_data, 
                                                          source_vocab_size,
                                                          encoding_embedding_size)
    # RNN cell
    def get_lstm_cell(rnn_size):
        lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return lstm_cell
    #cell = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(rnn_size) for _ in range(num_layers)])
    fw_cell = get_lstm_cell(rnn_size)
    bw_cell = get_lstm_cell(rnn_size)
    with tf.name_scope('encoder'):
    #    encoder_output, encoder_state = tf.nn.dynamic_rnn(cell, encoder_embed_input,
    #                                                sequence_length=source_sequence_length,dtype=tf.float32)
         (output_fw, output_bw), (encoder_fw_state, encoder_bw_state) = tf.nn.bidirectional_dynamic_rnn(fw_cell,bw_cell,encoder_embed_input,sequence_length=source_sequence_length,dtype=tf.float32)
         encoder_output = tf.concat([output_fw, output_bw],axis=2)
         encoder_state_c = tf.concat((encoder_fw_state[0],encoder_bw_state[0]),-1)
         encoder_state_h = tf.concat((encoder_fw_state[1],encoder_bw_state[1]),-1)
         encoder_state = tf.contrib.rnn.LSTMStateTuple(c = encoder_state_c, h = encoder_state_h)
         #encoder_output = output_fw
         #encoder_state = tf.concat([output_state_fw, output_state_bw],axis=1) 
#         encoder_states = []
#         for i in range(1):
#             if isinstance(encoder_fw_state[i], tf.contrib.rnn.LSTMStateTuple):
#                 encoder_state_c = tf.concat(values=(encoder_fw_state[i].c, encoder_bw_state[i].c),axis=1,name="encod_fw_state_c")
#                 encoder_state_h = tf.concat(values=(encoder_fw_state[i].h, encoder_bw_state[i].h),axis=1,name="encod_fw_state_h")
#                 encoder_state = tf.contrib.rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)
#             elif isinstance(encoder_fw_state[i], tf.Tensor):
#                 encoder_state = tf.concat(values=(encoder_fw_state[i], encoder_bw_state[i]),axis=1,name="bidirectional_concat")

#             encoder_states.append(encoder_state)

#    encoder_states = tuple(encoder_states)
        #tf.summary.histogram("encoder/out",encoder_output)
    return encoder_output,encoder_state




# decoder
# cut  <PAD> <EOS>
# <GO>
def process_decoder_input(target, vocab_to_int, batch_size):
    ending = tf.strided_slice(target, [0,0], [batch_size, -1], [1,1])
    decoder_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)
    return decoder_input




# Embedding
def decoding_layer(target_letter_to_int, decoding_embedding_size, num_layers,
                  rnn_size, target_sequence_length, max_target_sequence_length,
                  encoder_state, decoder_input,source_sequence_length, encoder_output):
    # 1 embedding
    target_vocab_size = len(target_letter_to_int)
    decoder_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    decoder_embed_input = tf.nn.embedding_lookup(decoder_embeddings, decoder_input)
    # 2 decode rnn
    def get_decoder_cell(rnn_size):
        decoder_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return decoder_cell
    
    cell = get_decoder_cell(rnn_size*2)
    

    # Create an attention mechanism
    attention_mechanism = tf.contrib.seq2seq.LuongAttention(
        rnn_size*2, encoder_output,
        memory_sequence_length=source_sequence_length)
    
    # attention wrapper
    a_cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism, attention_layer_size=rnn_size*2)
    
    # 3 output 
    output_layer = Dense(target_vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
    # 4 Training decoder
    with tf.variable_scope('decode'):
        # helper
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
                                                           sequence_length=target_sequence_length,
                                                           time_major=False)
        # decoder
        decoder_initial_state = a_cell.zero_state(batch_size, dtype=tf.float32).clone(cell_state=encoder_state)
        training_decoder = tf.contrib.seq2seq.BasicDecoder(a_cell, 
                                                           training_helper,
                                                           decoder_initial_state, 
                                                           output_layer)
        
        training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                      impute_finished=True,
                                                                      maximum_iterations=max_target_sequence_length)
        #tf.summary.tensor_summary('decoder_output',training_decoder_output)
    with tf.variable_scope('decode', reuse=True):
        
        start_tokens = tf.tile(tf.constant([target_letter_to_int['<GO>']], dtype=tf.int32), [batch_size], name='start_tokens')
        
        predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embeddings,
                                                                    start_tokens,
                                                                    target_letter_to_int['<EOS>'])
        
        predicting_decoder = tf.contrib.seq2seq.BasicDecoder(a_cell,
                                                               predicting_helper,
                                                               decoder_initial_state,
                                                               output_layer
                                                              )
        predicting_decoder_output,_,_ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
                                                                       impute_finished=True,
                                                                       maximum_iterations=max_target_sequence_length)
        return training_decoder_output, predicting_decoder_output




# seqseq encode decoder
def seq2seq_model(input_data, targets, lr, target_sequence_length, max_target_sequence_length,
                 source_sequence_length, source_vocab_size, target_vocab_size,
                 encoder_embedding_size, decoder_embedding_size,
                 rnn_size, num_layers):
    encoder_output, encoder_state = get_encoder_layer(input_data,rnn_size,num_layers,
                                        source_sequence_length,
                                        source_vocab_size,
                                        encoding_embedding_size)
    
    
    decoder_input = process_decoder_input(targets, target_letter_to_int, batch_size)
    
    
    training_decoder_output, predicting_decoder_output = decoding_layer(target_letter_to_int,
                                                                       decoding_embedding_size,
                                                                       num_layers,
                                                                       rnn_size,
                                                                       target_sequence_length,
                                                                       max_target_sequence_length,
                                                                       encoder_state,
                                                                       decoder_input,
                                                                       source_sequence_length,
                                                                       encoder_output)
    return training_decoder_output, predicting_decoder_output





epochs = 100
batch_size = 128
rnn_size = 50
num_layers = 2
encoding_embedding_size = 50
decoding_embedding_size = 50
learning_rate = 0.001





train_graph = tf.Graph()
with train_graph.as_default():
    
    input_data, targets, lr, target_sequence_length, max_target_sequence_length, source_sequence_length = get_inputs()
    training_decoder_output, predicting_decoder_output = seq2seq_model(input_data,
                                                                      targets,
                                                                      lr,
                                                                      target_sequence_length,
                                                                      max_target_sequence_length,
                                                                      source_sequence_length,
                                                                      len(source_letter_to_int),
                                                                      len(target_letter_to_int),
                                                                      encoding_embedding_size,
                                                                      decoding_embedding_size,
                                                                      rnn_size,
                                                                      num_layers)
    
    training_logits = tf.identity(training_decoder_output.rnn_output, 'logits')
    predicting_logits = tf.identity(predicting_decoder_output.sample_id, name='predictions')
    
    masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32,name='masks')
    
    with tf.name_scope('optimization'):
        cost = tf.contrib.seq2seq.sequence_loss(training_logits, targets, masks)
        tf.summary.scalar('cost',cost)
        optimizer = tf.train.AdamOptimizer(lr)
        
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_norm(grad, 5.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)
    init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
#train_graph.finalize()




# batches
def pad_sentense_batch(sentence_batch, pad_int):
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int]*(max_sentence - len(sentence)) for sentence in sentence_batch]




def get_batches(targets, sources, batch_size, source_pad_int, target_pad_int):
    for batch_i in range(0, len(sources)//batch_size):
        start_i = batch_i*batch_size
        source_batch = sources[start_i:start_i+batch_size]
        target_batch = targets[start_i:start_i+batch_size]
        
        pad_source_batch = np.array(pad_sentense_batch(source_batch, source_pad_int))
        pad_target_batch = np.array(pad_sentense_batch(target_batch, target_pad_int))
        
        target_length = []
        for target in target_batch:
            target_length.append(len(target))
        
        source_length = []
        for source in source_batch:
            source_length.append(len(source))
        
        yield pad_target_batch, pad_source_batch, target_length, source_length
            




# taind

train_source = source_int[batch_size:]
train_target = target_int[batch_size:]
valid_source = source_int[:batch_size]
valid_target = target_int[:batch_size]
(valid_targets_batch, valid_sources_batch, valid_targets_lengths, valid_sources_lengths) = next(get_batches(valid_target, valid_source, batch_size,
                           source_letter_to_int['<PAD>'],
                           target_letter_to_int['<PAD>']))
display_step = 4 

checkpoint = "trained_model.ckpt" 
config = tf.ConfigProto();
config.gpu_options.allocator_type = 'BFC';
config.gpu_options.allow_growth = True;
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True) #options = run_opts in session
with tf.Session(graph=train_graph,config=config) as sess:
    #merged = tf.summary.merge_all()
    #writer = tf.summary.FileWriter("logs/", sess.graph)
    sess.run(init_op)
        
    #options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    #run_metadata = tf.RunMetadata()
    for epoch_i in range(1, epochs+1):
        for batch_i, (targets_batch, sources_batch, targets_lengths, sources_lengths) in enumerate(
                get_batches(train_target, train_source, batch_size,
                           source_letter_to_int['<PAD>'],
                           target_letter_to_int['<PAD>'])):
            
            _, loss = sess.run(
                [train_op, cost],
                {input_data: sources_batch,
                 targets: targets_batch,
                 lr: learning_rate,
                 target_sequence_length: targets_lengths,source_sequence_length: sources_lengths})

            #result = sess.run([merged],
            #{input_data: sources_batch,
            #targets: targets_batch,
            #lr: learning_rate,
            #target_sequence_length: targets_lengths,source_sequence_length: sources_lengths})
            #writer.add_summary(result,batch_i)
            if batch_i % display_step == 0:
                
                #loss
                validation_loss = sess.run(
                [cost],
                {input_data: valid_sources_batch,
                 targets: valid_targets_batch,
                 lr: learning_rate,
                 target_sequence_length: valid_targets_lengths,
                 source_sequence_length: valid_sources_lengths})
                
                print('Epoch {:>3}/{} Batch {:>4}/{} - Training Loss: {:>6.3f}  - Validation loss: {:>6.3f}'
                      .format(epoch_i,
                              epochs, 
                              batch_i, 
                              len(train_source) // batch_size, 
                              loss, 
                              validation_loss[0]))


                #fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                #chrome_trace = fetched_timeline.generate_chrome_trace_format()
                #with open('timeline/'+str(epoch_i)+'timeline_02_step_%d.json' % batch_i, 'w') as f:
                #    f.write(chrome_trace)
        if epoch_i%5==0:
      
            saver = tf.train.Saver()
            saver.save(sess, "./paramatersbi/"+str(epoch_i)+checkpoint)
            print('Model Trained and Saved')


