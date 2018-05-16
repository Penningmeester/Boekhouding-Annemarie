import model_training
import predict
import Data_importer

from datetime import datetime as dt

def main():

	tstart = dt.now()
    model_training.main()
    print('Finished training model in:\n', dt.now()-tstart)
    print('\n predicting outcome of test set:')
    predict.main()
    print('\nFinished predicting in:\n',dt.now()-tstart)
	print('\nFinished whole program in:\n',dt.now()-tstart)

	print('Predictions can be found in:', Data_importer.location_lookup()["submission_path"])
if __name__=="__main__":
    main()