import Data_importer
import feature_engineering


def main():
	test = Data_importer.load_test_set()
	feature_engineering.main(test)
