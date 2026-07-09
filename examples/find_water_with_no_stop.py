from fairmofsyncondition.read_write import filetyper, coords_library

data = filetyper.load_data("../data/json_data/final_data_for_ML_training2.json")
for refcode in data:
    for temp in data[refcode]["reaction_temperature"]:
        if temp is not None and temp > 500:
            print(refcode, temp)
    # for salt in data[refcode]["metal_salt"]:
    #     if "H2O" in salt and "." not in salt:
    #         print(refcode, salt)