import csv
import os
import json

def add_radiomics_with_metadatapath_to_files, path_to_csv):
    file_list = os.listdir(path_to_files)
    count = 0

    # Write File
    with open('test_normal_viral_covid_added.csv', mode='w', newline='') as employee_file:
        employee_writer = csv.writer(employee_file, delimiter=',')
        # Read File
        with open(path_to_csv, 'r') as file:
            csv_reader = csv.reader(file, delimiter=',')
            for index, row in enumerate(csv_reader):
                values = row
                if index == 0:
                    values.extend(['Age', 'Sex', 'Temperature', 'ICU_admission', 'Intubation'])
                    employee_writer.writerow(values)
                else:
                    file_name = row[0].split(".png")[0] + ".json"
                    if file_name in file_list:
                        file_path = os.path.join(path_to_files, file_name)
                        # Read JSON
                        with open(file_path) as f:
                            data = json.load(f)
                            if len(data["annotations"]) > 3:
                                val_dict = {"Age": 0, "Sex": 0, "Temperature": 0.0, "ICU": 0, "Intb": 0}
                                #print("Reading FIle:", file_path)
                                for arg in data["annotations"]:
                                    if "polygon" not in arg.keys():
                                        try:
                                            # print(arg)
                                            if "Age" in arg["name"]:
                                                age = arg["name"].split("Age:")[-1]
                                                val_dict["Age"] = age
                                                # print("Age:", age)
                                            elif "Sex" in arg["name"]:
                                                sex = arg["name"].split("Sex/")[-1]
                                                if sex == "M":
                                                    val_dict["Sex"] = 1
                                                else:
                                                    val_dict["Sex"] = 2
                                                # print("Sex:", sex)
                                            elif "Temperature" in arg["name"]:
                                                temp = arg["name"].split("Temperature:")[-1]
                                                val_dict["Temperature"] = temp
                                                # print("Temperature:", temp)
                                            elif "ICU_admission" in arg["name"]:
                                                icu = arg["name"].split("ICU_admission/")[-1]
                                                if icu == "Y":
                                                    val_dict["ICU"] = 1
                                                else:
                                                    val_dict["ICU"] = 2
                                                # print("ICU:", icu)
                                            elif "Intubation_present" in arg["name"]:
                                                intb = arg["name"].split("Intubation_present/")[-1]
                                                if intb == "Y":
                                                    val_dict["Intb"] = 1
                                                else:
                                                    val_dict["Intb"] = 2
                                                # print("Intubation:", intb)
                                        except Exception as e:
                                            print(e)
                                # Write to dict
                                values.extend([val_dict["Age"], val_dict["Sex"], val_dict["Temperature"],
                                               val_dict["ICU"], val_dict["Intb"]])
                                # Write to file
                                employee_writer.writerow(values)
                            else:
                                values.extend([0, 0, 0.0, 0, 0])
                                # Write to file
                                employee_writer.writerow(values)
                    else:
                        print("File {} not found:", file_name)


if __name__ == "__main__":
    path_to_files = "path_to_annotations_file"
    path_to_csv = "path_to_csv_file_with_radiomics_features.csv"
    add_radiomics_with_metadata(path_to_files, path_to_csv)