#!/usr/bin/env python3
import os
import sys
import re
import argparse
import pickle
import torch
import glob
import warnings
import lmdb
import tarfile
import zipfile
import tempfile
import shutil
import gzip
from typing import Iterable, Union
import numpy as np
from ase import Atoms
from ase.io import read
from pathlib import Path
from torch_geometric.data import Data
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from mofstructure.structure import MOFstructure
from mofstructure import mofdeconstructor
from fairmofsyncondition.read_write import filetyper, coords_library

warnings.filterwarnings(
    "ignore",
    message="dict interface is deprecated. Use attribute interface instead",
    category=DeprecationWarning,
    module=r"pymatgen\.symmetry\.analyzer"
)

warnings.filterwarnings(
    "ignore",
    message=r"We strongly discourage using implicit binary/text `mode`",
    category=FutureWarning,
    module=r"pymatgen\.io\.cif"
)

convert_struct = {'cubic': 0,
                  'hexagonal': 1,
                  'monoclinic': 2,
                  'orthorhombic': 3,
                  'tetragonal': 4,
                  'triclinic': 5,
                  'trigonal': 6
                  }

def encode_peaks_with_hkl(pattern):

    hkls_groups = pattern.hkls
    d_list = pattern.d_hkls
    hkl_rows = []
    for group, d in zip(hkls_groups, d_list):
        print(group)
        h, k, l = group[0]["hkl"]
        mult_total = sum(ref["multiplicity"] for ref in group)
        # n_sym = len(group)
        hkl_rows.append([h, k, l, mult_total, d])
        # hkl_rows.append([h, k, l, mult_total, n_sym, d])
    # hklmd = torch.tensor(hkl_rows, dtype=torch.float32)
    # feat = torch.cat([xrd, hklmd], dim=-1)
    return hkl_rows


class DataStructure(object):
    def __init__(self, ase_atom=None, wavelength='CuKa'):
        if isinstance(ase_atom, Atoms):
            self.ase_atom = ase_atom
        elif isinstance(ase_atom, str) and os.path.isfile(ase_atom):
            self.ase_atom = read(ase_atom)
        elif isinstance(ase_atom, Data):
            self.ase_atom = coords_library.pytorch_geometric_to_ase(ase_atom)
        else:
            print("sorry input type is not recorgnised")
        self.structure = MOFstructure(self.ase_atom)
        self.pymat = AseAtomsAdaptor.get_structure(self.ase_atom)
        self.torch_data = coords_library.ase_to_pytorch_geometric(self.ase_atom)
        self.wavelength = wavelength

    def get_xrd(self):
        xrd_cal = XRDCalculator(wavelength=self.wavelength)
        pattern = xrd_cal.get_pattern(self.pymat, two_theta_range=(0, 100))
        two_theta = np.asarray(pattern.x, dtype=np.float32)
        intensity = np.asarray(pattern.y, dtype=np.float32)
        xrd = np.stack([two_theta, intensity], axis=-1)
        # hklmsd = encode_peaks_with_hkl(pattern)
        return xrd

    def convert_metals(self):
        '''
        return metals dict as a dictionary of symbols with index
        '''
        return {j: i for i, j in enumerate(mofdeconstructor.transition_metals()[1:])}

    def get_species_conc(self):
        """
        Create a one hot encodeing for the concentration of each
        atomic species found in the system. Similar to the empirical
        formular of the system.
        """
        emb = torch.zeros(120)
        atomic_num = self.ase_atom.get_atomic_numbers()
        a, b = np.unique(atomic_num, return_counts=True)
        for aa, bb in zip(a, b):
            emb[aa] = bb
        return emb

    def get_coordination_and_oms(self):
        """
        Get the oms embeding
        """
        general = self.structure.get_oms()
        metals = general['metals']
        tmp_dict = dict()
        emb = torch.zeros(96)
        for i in general["metal_info"]:
            cord = i["coordination_number"]
            metal = i["metal"]
            if metal in tmp_dict:
                if cord > tmp_dict[metal]:
                    tmp_dict[metal] = cord
            else:
                tmp_dict[metal] = cord

        # for i, j in tmp_dict.items():
        #     emb[self.convert_metals()[i]] = j
        metal_map = self.convert_metals()
        for metal, coord in tmp_dict.items():
            idx = metal_map.get(metal)
            if idx is None:
                continue
            emb[idx] = coord
        oms = general["has_oms"]
        return emb, oms, metals

    def get_space_group(self):
        "Get space group embedding"
        emb_sg = torch.zeros(231)
        emb_cs = torch.zeros(7)
        sga = SpacegroupAnalyzer(self.pymat)
        space_group_number = sga.get_space_group_number()
        emb_sg[space_group_number] = 1
        get_crystal_system = sga.get_crystal_system()
        emb_cs[convert_struct[get_crystal_system]] = 1
        return emb_sg, emb_cs, space_group_number, get_crystal_system

    def get_general_torch_data(self):
        emb_sg, emb_cs, _, _ = self.get_space_group()
        cn_emb, oms, _ = self.get_coordination_and_oms()
        atom_conc = self.get_species_conc()
        self.torch_data.atomic_one_hot = atom_conc
        self.torch_data.cordinates = cn_emb
        self.torch_data.space_group_number = emb_sg
        self.torch_data.crystal_system = emb_cs
        self.torch_data.oms = torch.tensor([[oms]], dtype=torch.float)
        return self.torch_data

    def complete_torch_data(self):
        '''
        Get torch data
        '''
        emb_sg, emb_cs, _, _ = self.get_space_group()
        cn_emb, oms, metals = self.get_coordination_and_oms()
        xrd = self.get_xrd()
        atom_conc = self.get_species_conc()
        self.torch_data.atomic_one_hot = atom_conc
        self.torch_data.cordinates = cn_emb,
        self.torch_data.space_group_number = emb_sg
        self.torch_data.crystal_system = emb_cs
        self.torch_data.oms = torch.tensor([[oms]], dtype=torch.float)
        self.torch_data.xrd = torch.from_numpy(xrd)
        # self.torch_data.hklmsd = torch.tensor(hkls, dtype=torch.float32)
        return self.torch_data



def save2lmdb(
    lmdb_path: Union[str, Path],
    cif_paths: Iterable[Union[str, Path]],
    map_size: int = 1 << 40,
    commit_interval: int = 1000
    ):
    """
    Save a collection of CIF-based PyTorch Geometric data objects into an LMDB,
    in a memory-efficient and query-friendly way.

    Keys stored in LMDB
    -------------------
    - b"data:{idx}"     -> pickled data object
    - b"idx:{idx}"      -> refcode as bytes (one refcode per index)
    - b"ref:{refcode}"  -> pickled list[int] of indices for that refcode
    - b"__len__"        -> pickled total count (int)
    """
    lmdb_path = str(lmdb_path)
    env = lmdb.open(lmdb_path, map_size=map_size)

    count = 0
    txn = env.begin(write=True)

    try:
        for cif_file in cif_paths:
            cif_file = str(cif_file)

            if not os.path.isfile(cif_file):
                print(f"[WARN] Not a file, skipping: {cif_file}")
                continue

            refcode = Path(cif_file).stem
            print(refcode)

            try:
                data = DataStructure(cif_file).complete_torch_data()

                required = ["x","edge_index","edge_attr","lattice","atomic_one_hot","cordinates","space_group_number","crystal_system","oms"]
                if not all(hasattr(data, k) for k in required):
                    continue

                # if len(data) != 10:
                #     print(f"[WARN] Data length != 10 for {refcode}, skipping")
                #     continue

                data_key = f"data:{count}".encode("ascii")
                blob = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
                txn.put(data_key, blob)

                idx_key = f"idx:{count}".encode("ascii")
                txn.put(idx_key, refcode.encode("utf-8"))

                ref_key = f"ref:{refcode}".encode("utf-8")
                existing = txn.get(ref_key)
                if existing is not None:
                    indices = pickle.loads(existing)
                else:
                    indices = []
                indices.append(count)
                txn.put(ref_key, pickle.dumps(indices, protocol=pickle.HIGHEST_PROTOCOL))

                count += 1

                if count % commit_interval == 0:
                    txn.commit()
                    txn = env.begin(write=True)

            except Exception as e:
                print(f"[ERROR] Skipping {refcode}: {e}")
                continue

        txn.put(b"__len__", pickle.dumps(count, protocol=pickle.HIGHEST_PROTOCOL))
        txn.commit()

    finally:
        env.close()

def collect_cif_files(input_path: str):
    """
    Return a sorted list of CIF files from a given file, directory,
    or compressed archive (.zip, .tar, .tar.gz, .tgz, .cif.gz).
    """
    p = Path(input_path)

    if not p.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if p.is_file():

        if p.suffix.lower() == ".cif":
            return [str(p)]

        if str(p).lower().endswith(".cif.gz"):
            out_path = p.parent / p.with_suffix("").name
            if not out_path.exists():
                with gzip.open(p, "rb") as f_in, open(out_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            return [str(out_path)]

        name_lower = p.name.lower()
        is_zip = name_lower.endswith(".zip")
        is_tar_like = (
            name_lower.endswith(".tar")
            or name_lower.endswith(".tar.gz")
            or name_lower.endswith(".tgz")
        )

        if is_zip or is_tar_like:
            extract_dir = p.parent / f"{p.stem}_extracted"
            extract_dir.mkdir(parents=True, exist_ok=True)

            if is_zip:
                with zipfile.ZipFile(p, "r") as zf:
                    zf.extractall(extract_dir)
            else:
                with tarfile.open(p, "r:*") as tf:
                    tf.extractall(extract_dir)

            cif_files = sorted(str(f) for f in extract_dir.rglob("*.cif"))
            if not cif_files:
                raise ValueError(f"No .cif files found in archive: {input_path}")
            return cif_files

        raise ValueError(f"Unsupported file type: {input_path}")

    if p.is_dir():
        cif_files = sorted(str(f) for f in p.glob("*.cif"))
        if not cif_files:
            raise ValueError(f"No .cif files found in directory: {input_path}")
        return cif_files

    raise FileNotFoundError(f"Input path is neither file nor directory: {input_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert CIF file(s) into a memory-efficient LMDB dataset."
    )

    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Path to a CIF file, archive (.zip/.tar/.tar.gz/.tgz/.cif.gz), or directory containing CIF files."
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output directory for LMDB database."
    )

    parser.add_argument(
        "--map-size",
        type=int,
        default=(1 << 40),
        help="LMDB maximum size in bytes (default: 1 TB)"
    )

    parser.add_argument(
        "--commit-interval",
        type=int,
        default=1000,
        help="Number of CIFs to process before each commit (default: 1000)"
    )

    args = parser.parse_args()

    try:
        cif_files = collect_cif_files(args.input)
    except Exception as e:
        parser.error(str(e))

    print(f"\nFound {len(cif_files)} CIF file(s).")
    print(f"Saving LMDB → {args.output}\n")

    save2lmdb(
        lmdb_path=args.output,
        cif_paths=cif_files,
        map_size=args.map_size,
        commit_interval=args.commit_interval,
    )

    print("\nLMDB database created successfully!")
    print(f" Location: {args.output}")
    print(f" Items stored (attempted): {len(cif_files)}")


if __name__ == "__main__":
    main()
