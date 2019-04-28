from keras.engine.topology import Layer, InputSpec
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Bidirectional, Embedding
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.models import load_model
from sklearn.metrics import precision_recall_fscore_support, classification_report, accuracy_score
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

seed = 1337
np.random.seed(seed)
# Import Files



class AttentionWithContext(Layer):
    """
        Attention operation, with a context/query vector, for temporal data.
        Supports Masking.
        Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
        "Hierarchical Attention Networks for Document Classification"
        by using a context vector to assist the attention
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(AttentionWithContext())
        """

    def __init__(self, init='glorot_uniform', kernel_regularizer=None, bias_regularizer=None, kernel_constraint=None, bias_constraint=None,  **kwargs):
        self.supports_masking = True
        self.init = initializers.get(init)
        self.kernel_initializer = initializers.get('glorot_uniform')

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight((input_shape[-1], 1),
                                 initializer=self.kernel_initializer,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)
        self.b = self.add_weight((input_shape[1],),
                                 initializer='zero',
                                 name='{}_b'.format(self.name),
                                 regularizer=self.bias_regularizer,
                                 constraint=self.bias_constraint)

        self.u = self.add_weight((input_shape[1],),
                                 initializer=self.kernel_initializer,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)
        self.built = True

    def compute_mask(self, input, mask):
        return None

    def call(self, x, mask=None):
        # (x, 40, 300) x (300, 1)
        multData =  K.dot(x, self.kernel) # (x, 40, 1)
        multData = K.squeeze(multData, -1) # (x, 40)
        multData = multData + self.b # (x, 40) + (40,)

        multData = K.tanh(multData) # (x, 40)

        multData = multData * self.u # (x, 40) * (40, 1) => (x, 1)
        multData = K.exp(multData) # (X, 1)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            mask = K.cast(mask, K.floatx()) #(x, 40)
            multData = mask*multData #(x, 40) * (x, 40, )

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        multData /= K.cast(K.sum(multData, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        multData = K.expand_dims(multData)
        weighted_input = x * multData
        return K.sum(weighted_input, axis=1)


    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1],)


def biLSTM(Xtrain, Ytrain, Xtest, Ytest, training, output, embeddings_index, tknzr, modelh5, cv, eps, ptc, ds, vb, bs, prob):
    if training:
        y_train_reshaped = to_categorical(Ytrain, num_classes=2)

        t = Tokenizer()
        t.fit_on_texts(Xtrain)
        vocab_size = len(t.word_index) + 1
        Xtrain = t.texts_to_sequences(Xtrain)
        max_length = max([len(s) for s in Xtrain + Xtest])
        X_train_reshaped = pad_sequences(Xtrain, maxlen=max_length, padding='post')
        with open(tknzr, 'wb') as handle:
            pickle.dump(t, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Padded the data')
        ## Loading in word embeddings and setting up matrix

        print('Loaded %s word vectors.' % len(embeddings_index))
        embedding_matrix = np.zeros((vocab_size, 200)) #Dimension vector in embeddings
        for word, i in t.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        print("Loaded embeddings")

        ### Setting up model
        embedding_layer = Embedding(vocab_size, 200, weights=[embedding_matrix], input_length=max_length, trainable=False, mask_zero=True)
        sequence_input = Input(shape=(max_length,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        l_lstm = Bidirectional(LSTM(512, return_sequences=True))(embedded_sequences)
        l_drop = Dropout(0.4)(l_lstm)
        l_att = AttentionWithContext()(l_drop)
        preds = Dense(2, activation='softmax')(l_att)
        model = Model(sequence_input, preds)
        model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['acc'])
        print(model.summary())
        print("Setting up model")


        ######## Preparing test data
        y_test_reshaped = to_categorical(Ytest, num_classes=2)

        X_test = t.texts_to_sequences(Xtest)
        X_test_reshaped = pad_sequences(X_test, maxlen=max_length, padding='post')
        print("Done preparing testdata")


        # filepath = modelh5
        checkpoint = ModelCheckpoint(modelh5, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=ptc)
        callbacks_list = [checkpoint, es]
        model.fit(X_train_reshaped, y_train_reshaped, epochs=eps, batch_size=bs, validation_data=(X_test_reshaped, y_test_reshaped), callbacks=callbacks_list, verbose=vb)
        loss, accuracy = model.evaluate(X_test_reshaped, y_test_reshaped, verbose=vb)

        print("Done training")

    if cv:
        # define 10-fold cross validation test harness https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        cvscores = []
        c = 0
        for train, test in kfold.split(Xtrain, Ytrain):
            c+=1
            y_train_reshaped = to_categorical(Ytrain, num_classes=2)

            t = Tokenizer()
            t.fit_on_texts(Xtrain)
            vocab_size = len(t.word_index) + 1
            Xtrain2 = t.texts_to_sequences(Xtrain)
            max_length = max([len(s) for s in Xtrain2 + Xtest])
            X_train_reshaped = pad_sequences(Xtrain2, maxlen=max_length, padding='post')
            with open(str(c)+'_'+tknzr, 'wb') as handle:
                pickle.dump(t, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # print('Padded the data')
            ## Loading in word embeddings and setting up matrix

            # print('Loaded %s word vectors.' % len(embeddings_index))
            embedding_matrix = np.zeros((vocab_size, 200)) #Dimension vector in embeddings
            for word, i in t.word_index.items():
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector

            ### Setting up model
            embedding_layer = Embedding(vocab_size, 200, weights=[embedding_matrix], input_length=max_length, trainable=False, mask_zero=True)
            sequence_input = Input(shape=(max_length,), dtype='int32')
            embedded_sequences = embedding_layer(sequence_input)
            l_lstm = Bidirectional(LSTM(512, return_sequences=True))(embedded_sequences)
            l_drop = Dropout(0.4)(l_lstm)
            l_att = AttentionWithContext()(l_drop)
            preds = Dense(2, activation='softmax')(l_att)
            model = Model(sequence_input, preds)
            model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['acc'])
            filepath = str(c)+'_'+modelh5
            checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=ptc)
            callbacks_list = [checkpoint,es]
            model.fit(X_train_reshaped[train], y_train_reshaped[train], epochs=eps, validation_data=(X_train_reshaped[test], y_train_reshaped[test]), batch_size=bs, callbacks=callbacks_list, verbose=vb)
            # evaluate the model
            scores = model.evaluate(X_train_reshaped[test], y_train_reshaped[test], verbose=vb)
            print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
            cvscores.append(scores[1] * 100)
        print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

    if not output:
        exit()

    if output:
        print("Loading tokenizer...")
        with open(tknzr, 'rb') as handle:
            t = pickle.load(handle)
        handle.close()
        print("Tokenizer loaded! Loading model...")
        model = load_model(modelh5, custom_objects={'AttentionWithContext': AttentionWithContext})
        print("Model loaded! Processing data...")

        # max_length = max([len(s) for s in Xtrain + Xtest])
        max_length = max([len(s) for s in Xtest])
        print('max_length: ', max_length)
        datalist_reshaped = t.texts_to_sequences(Xtest)
        try:
            datalist_reshaped = pad_sequences(datalist_reshaped, maxlen=max_length, padding='post')
            print("Data processed! Predicting values...")
            score = model.predict(datalist_reshaped)
            yguess = np.argmax(score, axis=1)

        except ValueError:
            try:
                 max_length = max([len(s) for s in Xtrain + Xtest])
                 datalist_reshaped = pad_sequences(datalist_reshaped, maxlen=max_length, padding='post')
                 print("Data processed! Predicting values...")
                 score = model.predict(datalist_reshaped)
                 yguess = np.argmax(score, axis=1)

            except ValueError:
                try:
                    datalist_reshaped = pad_sequences(datalist_reshaped, maxlen=851, padding='post')
                    print("Data processed! Predicting values...")
                    score = model.predict(datalist_reshaped)
                    yguess = np.argmax(score, axis=1)

                except ValueError:
                    datalist_reshaped = pad_sequences(datalist_reshaped, maxlen=2337, padding='post')
                    print("Data processed! Predicting values...")
                    score = model.predict(datalist_reshaped)
                    yguess = np.argmax(score, axis=1)
        
        yguess = [str(item) for item in yguess]
        Ytest = [str(item) for item in Ytest]

        # if prob:
            # with open('yguess_BiLSTM_' + ds + '.txt', 'w+') as yguess_output:
            #     for line in yguess:
            #         yguess_output.write('%s\n' % line)

        accuracy = accuracy_score(Ytest, yguess)
        precision, recall, f1score, support = precision_recall_fscore_support(Ytest, yguess, average="weighted")
        report = classification_report(Ytest, yguess)

        print("Predictions made! returning output")
        print(set(Ytest))
        print(set(yguess))
        return Ytest, yguess#, accuracy, f1score, report

    return True
