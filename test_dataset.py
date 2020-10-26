from dataset.RealEstate10K import RealEstate10K


dataset = RealEstate10K("D:\\MSI_NB\\source\\data\\RealEstate10K\\test")

for data in dataset:
    print(data)
