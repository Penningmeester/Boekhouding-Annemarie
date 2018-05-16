import model_training
import predict

from datetime import datetime as dt

def main():
	tstart = dt.now()
    model_training.main()
    print('Finished training model in:\n', dt.now()-tstart)
    print('\n predicting outcome of test set:')
    predict.main()
    print('\nFinished predicting in:\n',dt.now()-tstart)
	print('\nFinished whole program in:\n',dt.now()-tstart)
if __name__=="__main__":
    main()