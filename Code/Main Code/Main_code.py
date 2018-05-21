import model_training
import predict
import Data_importer

from datetime import datetime as dt

def main():

    tstart = dt.now()
    print('Starting with program\n')
    model_training.main()
    print('Finished training model in:\n', dt.now()-tstart)
    print('\n predicting outcome of test set:')
    bool_ = input('start predicting? Y/N?')
    if bool_=='Y':
        tstart2 = dt.now()
        predict.main()
        print('\nFinished predicting in:\n',dt.now()-tstart2)
    print('\nFinished whole program in:\n',dt.now()-tstart)

    print('Predictions can be found in:', Data_importer.location_lookup()["submission_path"])
if __name__=="__main__":
    main()