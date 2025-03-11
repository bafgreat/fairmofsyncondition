from fairmofsyncondition.read_write import  coords_library

path_to_mdb = "../data/lmdb_data/mof_syncondition_data.lmdb"

data = coords_library.LMDBDataset(lmdb_path=path_to_mdb)
print(data[1])