
import os
import sys
module = os.path.abspath("C:/AmalSujith\\18145\\development\\wesad_experiments\\src\\main")
if module not in sys.path:
    sys.path.append(module)
from DataHandler import DataHandler

class ExampleRun:
    
    handler = DataHandler()
        
    print("Preparing data for model creation..")
    handler.load_all()
    print("Considering baseline experiment data:")
    handler.compute_features()
    print("Considering stress experiment data:")
    handler.compute_features_stress()
    
    batch_size = 4
    epochs = 1
    print("===============================================================")
    
    print("Creating the LSTM network with", epochs, "epochs.")
    # compute for one epoch
    (model, X_train, X_test, y_train, y_test) = \
        handler.create_network(epochs, batch_size)
        
    print("===============================================================")
    
    print("Evaluating LSTM network with", epochs, "epochs.")
    handler.get_model_results(model, X_train, X_test, y_train, y_test, batch_size)
    
    
    print("===============================================================")
    
    print("Loading and evaluating LSTM network from 5 epochs")
    # then load a previously computed 5 epoch model and display the results
    model_5_epochs = handler.load_model('model-2019-07-1310_40_14.h5')
    handler.get_model_results(model_5_epochs, X_train, X_test, y_train, y_test )

