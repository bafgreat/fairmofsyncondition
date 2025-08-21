from fairmofsyncondition.read_write import  coords_library

path_to_mdb = "../data/lmdb_data/mof_syncondition_data.lmdb"

all_data = coords_library.LMDBDataset(lmdb_path=path_to_mdb)

# for data in all_data:
#     print(data)

print(len(all_data[0]))