from __future__ import annotations
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import numpy.typing as npt
from os import PathLike
from pathlib import Path


class Atom:
    """Class containing the information of a single atom.
    
    Attributes
    ----------
    element : str
        The atomic symbol of the element.
    xyz : NDArray
        The x-, y-, and z-coordinates of the atom.
    """

    def __init__(
        self,
        element: str,
        xyz: npt.NDArray,
    ):
        self.element = element
        self.xyz = np.array(xyz, dtype=np.float64)


    def __repr__(self):
        return (
            f"{"Element":12}" f"{"X":11}" f"{"Y":11}" f"{"Z":11}\n"
            f"{self.element:9}"f"{self.xyz[0]:11.6f}"f"{self.xyz[1]:11.6f}"f"{self.xyz[2]:11.6f}\n"
        )
    

class Geometry:
    """Class storing the geometry of a single molecule.
    
    Attributes
    ----------
    atoms : list[Atom]
        A list of the atoms in the molecule.
    charge : int, default=0
        The total formal charge of the molecule.
    """

    def __init__(
        self,
        atoms: list[Atom],
        charge: int = 0,
    ):
        self.atoms = atoms
        self.charge = charge


    def get_num_atoms(self) -> int:
        return len(self.atoms)


    def get_coords(self) -> npt.NDArray:
        return np.array([i.xyz for i in self.atoms])


    def get_elements(self) -> list[str]:
        return [i.element for i in self.atoms]


    def new_xyz(self, new_xyzs: npt.NDArray):
        for i in range(len(self.atoms)):
            self.atoms[i].xyz = new_xyzs[i]


    def remove_atom(self, index: int):
        del self.atoms[index]


    def pop_atom(self, index: int) -> Atom:
        return self.atoms.pop(index)


    def add_atom(self, atom: Atom, index: int = None):
        if index is not None:
            self.atoms.insert(index, atom)
        else:
            self.atoms.append(atom)


    @classmethod
    def from_xyz(cls, file_path: str, charge: int = 0):
        """Read in XYZ file format and return atomic symbols and coordinates

        Parameters
        ----------
        file_path : str
            Full path to an XYZ file.

        Returns
        -------
        Geometry object
        """

        molecule_xyz = []

        with open(file_path) as f:
            for line in f:
                # Takes each line in the file minus the last character, which is just the \n
                line = line[:-1].split()
                if line:
                    # If the line isn't empty then append to the molecule_xyz
                    molecule_xyz.append(line)

        elements = [str(i[0]) for i in molecule_xyz[2:]]
        xyzs = np.array([[float(j) for j in i[1:]] for i in molecule_xyz[2:]])

        atoms = []

        for i, element in enumerate(elements):
            atoms.append(Atom(element, xyzs[i]))

        return Geometry(atoms, charge)


    @classmethod
    def from_smiles(cls, smiles_string: str):
        """Convert a SMILES string to a 3D molecule.

        Parameters
        ----------
        smiles_string : str
            Any valid SMILES string (for example, "N#Cc1nn[nH]c(C#N)1")

        Returns
        -------
        Geometry object
        """

        molecule = Chem.AddHs(Chem.MolFromSmiles(smiles_string))
        AllChem.EmbedMolecule(molecule, AllChem.ETKDGv3())
        charge = Chem.rdmolops.GetFormalCharge(molecule)

        xyz_string = Chem.rdmolfiles.MolToXYZBlock(molecule)
        molecule_xyz = [i.split() for i in xyz_string.split("\n")[2:-1]]

        elements = [str(i[0]) for i in molecule_xyz]
        xyzs = np.array([[float(j) for j in i[1:]] for i in molecule_xyz])

        atoms = []

        for i, element in enumerate(elements):
            atoms.append(Atom(element, xyzs[i]))

        return Geometry(atoms, charge)


    @classmethod
    def from_list(cls, elements: list[str], xyzs: npt.NDArray, charge: int = 0):
        if len(elements) != len(xyzs):
            raise ValueError("The list of elements and coordinates must be of the same size!")
        else:
            atoms = []
            for i, element in enumerate(elements):
                atoms.append(Atom(element, xyzs[i]))

        return Geometry(atoms, charge)


    def to_xyz(self, name: str, xyz_dir: PathLike | None = None):
        if xyz_dir is None:
            xyz_dir = Path.cwd()

        with open(xyz_dir/Path(f"{name}.xyz"), "w", newline="") as xyz_file:
            xyz_file.write(f"{self.get_num_atoms()}\n\n")
            for atom in self.atoms:
                xyz_file.write(f"{atom.element:3}{atom.xyz[0]:12.6f}{atom.xyz[1]:12.6f}{atom.xyz[2]:12.6f}\n")


    def __repr__(self):
        self_repr = f"{"Element":12}{"X":11}{"Y":11}{"Z":11}\n\n"
        for i in self.atoms:
            self_repr += f"{i.element:9}{i.xyz[0]:11.6f}{i.xyz[1]:11.6f}{i.xyz[2]:11.6f}\n"
        return self_repr


    def __add__(self, structure: Geometry):
        s1 = self.atoms
        s2 = structure.atoms
        return Geometry(s1+s2, self.charge+structure.charge)


    def __iter__(self):
        yield from self.atoms

    #def write_orca_input(self, name: str, input_dir: PathLike = Path("./"), **kwargs):


class Molecule:
    """Class that contains all molecule data.

    Attributes
    ----------
    smiles_string : str
        Any valid SMILES string.
    structure : Geometry
        The structure of the molecule
    molecule_name : str, optional
        The molecule name that will be used for naming output XYZ and INP files.
    protonated_atom_index : int, optional
        The line index of the atom that carries an extra proton (the parent charge site).
    neighboring_atom_index : int, optional
        The line index of an atom attached to the parent charge site.
    proton_position_indices : list of int, optional
        A list of the line indexes for each proton attached to the protonated atom.
    charge : int, default=1
        The charge of the molecule.

    Methods
    -------
    get_proton_position(attempt_number)
        Return the line number of the proton of interest.
    get_protonated_atom_position()
        Return the XYZ coordinates of the parent charge site.
    calculate_charge(smiles)
        Calculate the charge of a molecule denoted by a SMILES string
    """

    def __init__(
        self,
        smiles_string: str,
        structure: Geometry = None,
        molecule_name: str = None,
        protonated_atom_index: int = None,
        neighboring_atom_index: int = None,
        proton_position_indices: list[int] = None,
    ):
        self.smiles_string = smiles_string

        if structure is not None:
            self.structure = structure
        else:
            self.structure = Geometry.from_smiles(self.smiles_string)

        self.molecule_name = molecule_name if molecule_name is not None else "bean"

        if protonated_atom_index is not None or self.structure.charge == 0:
            self.protonated_atom_index = protonated_atom_index
        else:
            self.protonated_atom_index = (
                Molecule.substructure_match(self.smiles_string, "protonated_atom")
            )

        if neighboring_atom_index is not None or self.structure.charge == 0:
            self.neighboring_atom_index = neighboring_atom_index
        else:
            self.neighboring_atom_index = (
                Molecule.substructure_match(self.smiles_string, "neighbor")
            )

        if proton_position_indices is not None or self.structure.charge == 0:
            self.proton_position_indices = proton_position_indices
        else:
            self.proton_position_indices = (
                Molecule.substructure_match(self.smiles_string, "protons")
            )

        if self.structure.charge > 0:
            self.structure.new_xyz(
                self.structure.get_coords()
                - self.structure.get_coords()[self.protonated_atom_index]
            )


    def get_proton_index(self, attempt_number: int) -> int:
        return self.proton_position_indices[attempt_number]


    def get_num_atoms(self) -> int:
        """Get the number of atoms in the molecule."""
        return self.structure.get_num_atoms()


    def get_coords(self) -> npt.NDArray:
        return self.structure.get_coords()


    def get_elements(self) -> list[str]:
        return self.structure.get_elements()


    def get_charge(self) -> int:
        return self.structure.charge


    def remove_atom(self, index: int):
        _ = self.structure.remove_atom(index)


    def pop_atom(self, index: int):
        return self.structure.pop_atom(index)


    def add_atom(self, atom: Atom, index: int = None):
        self.structure.add_atom(atom, index)


    def get_proton_position(self, attempt_number: int) -> npt.NDArray:
        """Get the coordinates for the proton requested.

        Parameters
        ----------
        attempt_number : {0, 1, 2}
            The current attempt number.

        Returns
        -------
        proton_position: NDArray
            The coordinates of the requested proton.
        """
        return self.get_coords()[self.get_proton_index(attempt_number)]


    def get_protonated_atom_position(self) -> npt.NDArray:
        """Get the XYZ coordinates of the parent charge site."""
        return self.get_coords()[self.protonated_atom_index]


    @staticmethod
    def calculate_charge(smiles: str):
        return Chem.rdmolops.GetFormalCharge(Chem.MolFromSmiles(smiles))
    

    @staticmethod
    def substructure_match(smiles_string: str, match_type: str) -> tuple[int, list[int], int]:
        """Generate an RDKit molecule and search the structure for protonated atoms.

        Parameters
        ----------
        smiles_string : str
            Any valid SMILES string.
        match_type : {"protons", "protonated_atom", "neighbor"}
            Feature in the structure to match to.

        Returns
        -------
        protonated_atom_index : int
            The line number of the atom with an extra proton (the parent charge site).
        proton_position_indices : list[int]
            A list containing any protons attached to the parent charge site.
        near_neighbor_index : int
            The line number of an atom directly connected to the parent charge site.

        Notes
        -----
        The features available for matching are `"protons"`, `"protonated_atom"`, and `"neighbor"`

        `"protons"` will search for any protons attached to a protonated atom, e.g. R-NH3+, R-OH2+, and return the
        line indexes of each proton found.

        `"protonated_atom"` will search for the protonated atom itself, and return its line number. If there are multiple
        protonated atoms, it will return the first one.

        `"neighbor"` will search for the neighboring atoms to the protonated atom, except for the hydrogens, and return
        the line index of one of the neighbors.
        """

        molecule = Chem.AddHs(Chem.MolFromSmiles(smiles_string))
        params = AllChem.ETKDGv3()
        AllChem.EmbedMolecule(molecule, params)

        if match_type == "protons":
            proton_atom = Chem.MolFromSmarts(
                "[$([#1][#7H+]),$([#1][#7H2+]),$([#1][#7H3+]),$([#1][#8H+]),$([#1][#8H2+])]"
            )
            proton_position = molecule.GetSubstructMatches(proton_atom)

            proton_position_indices: list[int] = [i[0] for i in proton_position]
            return proton_position_indices

        elif match_type == "protonated_atom":
            charged_atom = Chem.MolFromSmarts("[#7H+,#7H2+,#7H3+,#8H+,#8H2+]")
            parent_position = molecule.GetSubstructMatches(charged_atom)

            protonated_atom_index: int = parent_position[0][0]
            return protonated_atom_index

        elif match_type == "neighbor":
            near_neighbor = Chem.MolFromSmarts(
                "[$([*][#7H+]),$([*][#7H2+]),$([*][#7H3+]),$([*][#8H+]),$([*][#8H2+])]"
            )
            near_neighbor = molecule.GetSubstructMatches(near_neighbor)
        
            near_neighbor_index: int = near_neighbor[0][0]
            return near_neighbor_index
        
        else:
            raise ValueError('Invalid Match Type! Please select from "protons", "protonated_atom", "neighbor"')