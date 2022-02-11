import src.data.make_dataset as md
import src.Datahandler.prepare_data as prep

filepath = "data/raw/elspotprices_7.2.2022.dk1.csv"

def main():
    data = md.clean_data(filepath)
    train_scaled, validation_scaled, test_scaled = prep.data_prep(data)
    print("hi")


if __name__ == '__main__':
    main()
