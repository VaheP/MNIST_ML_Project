from data_handler import DataHandler
from models import RandomForestModel
from trainer import Trainer
import config


def main():
    data_handler = DataHandler(config.DATA_PATH, config.DATA_PATH_TEST)
    data_handler.load_data()
    data_handler.split_data()

    model = RandomForestModel(**config.MODEL_PARAMS)
    trainer = Trainer(model)

    trainer.train(data_handler.X_train, data_handler.y_train)
    accuracy = trainer.evaluate(data_handler.X_test, data_handler.y_test)

    print(f'Model accuracy: {accuracy}')


if __name__ == '__main__':
    main()
