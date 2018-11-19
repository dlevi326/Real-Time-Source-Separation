from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D, UpSampling1D, Activation, LeakyReLU, PReLU, Dropout
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.models import load_model
from keras import backend as K
import numpy as np
from keras.callbacks import ModelCheckpoint, History, EarlyStopping
from tqdm import tqdm

def create_model(input_shape, num_filters_per_hidden_layer, filter_size_per_hidden_layer, filter_size_output_layer ):
	
	K.clear_session()
	
	model = Sequential()
	
	model.add(Conv1D(filters = num_filters_per_hidden_layer[0], kernel_size = filter_size_per_hidden_layer[0], padding='same', input_shape = input_shape))
	model.add(BatchNormalization())
	model.add(PReLU())
	
	for num_filters, filter_size in zip(num_filters_per_hidden_layer[1:], filter_size_per_hidden_layer[1:]): 
		model.add(Conv1D(filters = num_filters, kernel_size = filter_size, padding='same'))
		model.add(BatchNormalization())
		model.add(PReLU())


	model.add(Conv1D(1, kernel_size = filter_size_output_layer, padding='same'))
	
	return(model)
	
def train_model( model, train_inputs, train_labels, epochs, batch_size, validation_inputs, validation_labels, filepath, patience ):
	
	model.compile(optimizer = 'adam', loss='mean_squared_error')
	
	checkpointer = ModelCheckpoint(filepath = filepath, monitor = "val_loss", verbose = 1, mode = 'min', save_best_only = True)
	early_stopping = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = patience, verbose = 1, mode='auto')

	history = model.fit(	train_inputs, train_labels,
            				epochs = epochs,
                			batch_size = batch_size,
                			shuffle = True,
                			validation_data = (validation_inputs, validation_labels),
                			callbacks = [checkpointer, early_stopping])
    
	model = load_model(filepath)

	return(model, history)

def train_model_finetune( model, train_inputs, train_labels, epochs, batch_size ):
	
	model.compile(optimizer = 'adam', loss='mean_squared_error')
	
	history = model.fit(	train_inputs, train_labels,
            				epochs = epochs,
                			batch_size = batch_size,
                			shuffle = True)
    
	return(model, history)

def save_model( model, save_path ):
	model.save(save_path)

def load_model_( load_path ):
	model = load_model( load_path )
	return(model)

def predict_model( model, inputs ):
	predictions = model.predict(inputs, batch_size = None, verbose=0, steps=None)
	return(predictions)
	
def get_output( model, new_input ):
	get_output = K.function([model.layers[0].input, K.learning_phase()],
                                  	  [model.layers[ len(model.layers) - 1 ].output])
	layer_output = get_output([new_input, 0])[0]
	
	return(layer_output)
	
def get_output_multiple_batches(model, input_frames, batch_size = 100):
	
	batches_output_frames_holder = []
	for i in tqdm(range(0, input_frames.shape[0], batch_size)):
		batch_input_frames = input_frames[ i:i+batch_size , :, : ]
		batch_output_frames = get_output( model, batch_input_frames )
		batch_output_frames = np.reshape(batch_output_frames, (batch_output_frames.shape[0], -1)) 
		batches_output_frames_holder.append(batch_output_frames)
		
	output_frames_concatenated = np.concatenate( batches_output_frames_holder, axis = 0 )
	
	return(output_frames_concatenated)
	
def summary_statistics( filename, model_name, history, frame_time, snr_db, 
						num_filters_per_hidden_layer, filter_size_per_hidden_layer, filter_size_output_layer,
						epochs, batch_size):
	
	
	best_val_loss = min( history.history["val_loss"] )				
	best_epoch_index = history.history["val_loss"].index( best_val_loss )
	best_train_loss = history.history["loss"][ best_epoch_index ]
	
	print( "\tFCNN Name: " + model_name )
	print( "\tNumber of Filters Per Hidden Layer: " + ', '.join(map(str, num_filters_per_hidden_layer) )) 
	print( "\tFilter Size Per Hidden Layer: " + ', '.join(map(str, list(np.array(filter_size_per_hidden_layer)*1000)))  + " ms" )
	print( "\tFilter Size for Output Layer: " + str( filter_size_output_layer*1000 ) + " ms")
	print( "\tFrame Time: " + str( frame_time*1000 ) + " ms")
	print( "\tTotal Epochs: " + str(epochs) )
	print( "\tBatch Size: " + str(batch_size) + " examples" )
	print( "\tSNR: " + str( snr_db ) + " dB")
	print( "\tBest Epoch: " + str(  best_epoch_index + 1 ) )
	print( "\tTraining Loss: " + str( best_train_loss ) )
	print( "\tValidation Loss: " + str( best_val_loss ) )
	print("\n")
	with open(filename, "a") as text_file:
		text_file.write( "FCNN Name: " + model_name )
		text_file.write( "\n" )
		text_file.write( "Number of Filters Per Hidden Layer: " + ', '.join(map(str, num_filters_per_hidden_layer) )) 
		text_file.write( "\n" )
		text_file.write( "Filter Size Per Hidden Layer: " + ', '.join(map(str, list(np.array(filter_size_per_hidden_layer)*1000)))  + " ms" )
		text_file.write( "\n" )
		text_file.write( "Filter Size for Output Layer: " + str( filter_size_output_layer*1000 ) + " ms")
		text_file.write( "\n" )
		text_file.write( "Frame Time: " + str( frame_time*1000 ) + " ms")
		text_file.write( "\n" )
		text_file.write( "Total Epochs: " + str(epochs) )
		text_file.write( "\n" )
		text_file.write( "Batch Size: " + str(batch_size) + " examples" )
		text_file.write( "\n" )
		text_file.write( "SNR: " + str( snr_db ) + " dB")
		text_file.write( "\n" )
		text_file.write( "Best Epoch: " + str(  best_epoch_index + 1 ) )
		text_file.write( "\n" )
		text_file.write( "Training Loss: " + str( best_train_loss ) )
		text_file.write( "\n" )
		text_file.write( "Validation Loss: " + str( best_val_loss ) )
		text_file.write( "\n\n" )