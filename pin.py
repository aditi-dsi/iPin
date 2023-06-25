import pandas as pd
import pickle
import numpy as np

dataset = pd.read_csv('PinDataset.csv', header=0, encoding='unicode_escape')
dataset.columns = dataset.columns.str.strip()

""" Function for PINCODE --> DISTRICT """

with open('iModelv2.pkl', 'rb') as file:
    model = pickle.load(file)


def get_only_district(pincodes):
    pincodes = np.array([int(pincode) for pincode in pincodes])
    pincodes = pincodes.reshape(-1, 1)
    output = model.predict(pincodes)
    return output


""" Function for PINCODE --> VILLAGES """


def pin_to_village(pincodes):
    output = {}
    for pincode in pincodes:
        pincode = int(pincode.strip())
        filtered_data = dataset.loc[dataset['Pincode'] == pincode]
        if not filtered_data.empty:
            villages = filtered_data['Village'].values
            output[pincode] = list(villages)
        else:
            print(f"No village found for Pincode: {pincode}")
    return output


""" Function for PINCODE --> SUB_DISTRICT + POST_OFFICE + DISTRICT """


def get_by_pincode(pincodes):
    output = {}
    for pincode in pincodes:
        pincode = int(pincode.strip())
        filtered_data = dataset.loc[dataset['Pincode'] == pincode]
        if not filtered_data.empty:
            tehsils = filtered_data['Tehsil'].unique()
            postoffices = filtered_data['Officename'].values
            districts = filtered_data['District'].values
            output[pincode] = list(zip(tehsils, postoffices, districts))
        else:
            print(f"No values found for Pincode: {pincode}")
    return output


""" Function for VILLAGE --> PINCODE + DISTRICT """


def get_by_value(villages):
    output = {}
    for village in villages:
        village = village.strip().lower()  # Convert to lowercase and remove whitespace
        filtered_data = dataset.loc[dataset['Village'].str.strip().str.lower() == village]
        if not filtered_data.empty:
            pincodes = filtered_data['Pincode'].values
            districts = filtered_data['District'].values
            output[village] = list(zip(pincodes, districts))
        else:
            print(f"No pincode values found for village: {village}")
    return output


""" For handling user requests by mapping them to respective functions """


def main():
    while True:
        print("Choose\n--> 1 - for pincode to district\n--> 2 - for pincode to "
              "village\n--> 3 - for pincode to all\n--> 4 - for Village to"
              "pincode\n--> 0 - to exit\n")
        num = int(input("Enter your choice:"))

        if num == 1:
            entry1 = input("Enter Pincodes (comma-separated): ")
            entry1 = entry1.split(',')
            print("Loading..")
            result = get_only_district(entry1)
            print(*result, sep='\n')
        elif num == 2:
            entry2 = input("Enter Pincodes (comma-separated): ")
            entry2 = entry2.split(',')
            print("Loading..")
            result = pin_to_village(entry2)
            for pincode, data in result.items():
                print(f"\nPincode: {pincode}\n")
                print(*data, sep='\n')
        elif num == 3:
            entry3 = input("Enter Pincodes (comma-separated): ")
            entry3 = entry3.split(',')
            print("Loading..")
            result = get_by_pincode(entry3)
            for pincode, data in result.items():
                print(f"Pincode: {pincode}")
                for tehsil, postoffice, district in data:
                    print(f"Teh: {tehsil}, PO: {postoffice}, Dist: {district}\n")
        elif num == 4:
            entry4 = input("Enter Villages (comma-separated): ")
            entry4 = entry4.split(',')
            print("Loading..")
            result = get_by_value(entry4)
            for village, data in result.items():
                print(f"Village: {village}")
                for pincode, district in data:
                    print(f"Pincode: {pincode}, District: {district}\n")
        elif num == 0:
            break
        else:
            print("Please enter a valid input")


""" To Run """

if __name__ == '__main__':
    main()
