import Data_importer
import feature_engineering


def main():
	test = Data_importer.load_test_set()
	feature_engineering.main(test)

	#predict if customer clicks
	print('Loading Click model')
	model = Data_importer.load_model(False)
	print('Predicting probability of Click')

	click_predict = model.predict_proba(test.values)[;,1]
	click_predict = list(click_predict*-1)
	print('Predicted Clicking probability')

	print('Loading Book model')
	model = Data_importer.load_model(True)
	print('Predicting probability of Booking')

	book_predict = model.predict_proba(test.values)[;,1]
	book_predict = list(book_predict*-1)
	print('Predicted the Booking probability')

	#Wrinting the predictions
	predictions = zip(test["srch_id"], test["prop_id"], 4*book_predict+click_predict)
	print("Writing predictions to file..")
    Data_importer.write_submission(recommendations)
	print("wrote predictions to file")