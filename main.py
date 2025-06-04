from __future__ import annotations
import argparse
import platform
import numpy as np
import numpy.typing as npt
import csv
from tkinter import Tk
from tkinter import filedialog
from os import PathLike, path
from pathlib import Path
from linear_algebra import LinearAlgebra
from linear_algebra import IDENTITY_MATRIX, Z_UNIT_VECTOR, Z_REFLECTOR, Z_ROTATION_180
from molecule import Atom, Geometry, Molecule

if platform.system() == "Windows":
    from ctypes import windll
    windll.user32.SetProcessDPIAware()


def structure_one_generator(
    molecule_alignment_quaternion: npt.NDArray,
    mol: Molecule,
) -> Geometry:
    """Align a molecule along Z axis and return the structure"""

    new_coords = []

    for atom_position in mol.get_coords():
        new_coord = np.dot(molecule_alignment_quaternion, atom_position)
        new_coords.append(new_coord)

    structure_one = Geometry.from_list(mol.get_elements(), new_coords)

    return structure_one


def structure_two_generator(
    stage: str,
    z_rotation_offset: npt.NDArray,
    structure_one: Geometry,
    distance_multiplier: float = 2.7,
    z_flipper: npt.NDArray = Z_REFLECTOR,
    disable_nudge: bool = False,
) -> Geometry:
    """Generate the second structure that the proton will be transferred to.

    Parameters
    ----------
    stage : {"Reactant", "Product", "Transition"}
        Stage of the reaction.
    z_rotation_offset : NDArray
        A matrix that rotates around the Z-axis.
    structure_one : ArrayLike
        The first structure generated from `structure_one_generator()`.
    distance_multiplier : float, default=2.7
        The distance between the parent charge sites in Angstroms.

    Returns
    -------
    structure_two : list[list[float]]
        The second structure to which the proton will be transferred.
    """

    if disable_nudge:
        nudge_rotate_x = IDENTITY_MATRIX
        nudge_rotate_y = IDENTITY_MATRIX
        nudge_translate = np.array([0., 0., 0.])

    else:
        nudge_rotate_x, nudge_rotate_y, nudge_translate = LinearAlgebra.nudge_matrix_generator(stage)

    new_coords = []

    for atom_position in structure_one.get_coords():
        if stage == "Reactant":
            new_coord = np.dot(
                nudge_rotate_x,
                (
                    np.dot(
                        nudge_rotate_y,
                        np.dot(
                            (
                                (np.dot(z_flipper, atom_position))
                                + (distance_multiplier * Z_UNIT_VECTOR)
                            ),
                            z_rotation_offset
                        )
                    )
                ),
            ) + (nudge_translate)
        elif stage == "Product":
            new_coord = np.dot(
                nudge_rotate_x,
                (
                    np.dot(
                        nudge_rotate_y,
                        np.dot(
                            (
                                (np.dot(z_flipper, atom_position))
                                + (distance_multiplier * Z_UNIT_VECTOR)
                            ),
                            z_rotation_offset
                        )
                    )
                ),
            ) + ((-1) * nudge_translate)
        elif stage == "Transition":
            new_coord = np.dot(
                z_rotation_offset,
                (
                    np.dot(z_flipper, atom_position)
                    + ((distance_multiplier - 0.1) * Z_UNIT_VECTOR)
                )
            )
        new_coords.append(new_coord)

    structure_two = Geometry.from_list(structure_one.get_elements(), new_coords)

    return structure_two


def structure_checker(
    structure_one: Geometry,
    structure_two: Geometry,
    mol: Molecule,
    attempt_number: int,
) -> bool:
    """Check for overlap between the two molecules

    Parameters
    ----------
    structure_one : ArrayLike
        The structure generated from `structure_one_generator()`.
    structure_two : ArrayLike
        The structure generated from `structure_two_generator()`.
    mol : Molecule
        A Molecule object
    attempt_number : int
        The attempt number for this run, typical maximum is 2 (3 possible tries)

    Returns
    -------
    bool
        True indicates that the structures have some overlap 
        (atoms within 1.5 Angstroms of each other).
        False indicates there is no detected overlap.
    """

    n = mol.get_proton_index(attempt_number)
    for i, atom_one in enumerate(structure_one):
        for j, atom_two in enumerate(structure_two):
            if (i == n) or (j == n - 1):
                continue
            # Only executes if not proton involved in transport
            if LinearAlgebra.distance_calculator(atom_one.xyz, atom_two.xyz) < 1.5:
                return True
            else:
                continue
    # We checked all atoms, none overlapped
    return False


def final_structure_generator(
    structure_one: Geometry,
    structure_two: Geometry,
    stage: str,
    mol: Molecule,
    attempt_number: int,
    distance_multiplier: float = 2.7,
) -> Geometry:
    """Generate the complete structure. Should only be used after checking for/handling overlap.

    Parameters
    ----------
    structure_one : Geometry
        The structure generated from `structure_one_generator()`.
    structure_two : Geometry
        The structure generated from `structure_two_generator()`.
    stage : {"Reactant", "Product", "Transition"}
        Stage of the reaction..
    mol : Molecule
        The current Molecule object.
    attempt_number : {0, 1, 2}
        The attempt number.
    distance_multiplier : float, default=2.7
        The distance between the parent charge sites in Angstroms.

    Returns
    -------
    final_structure : Geometry
        The final assembled structure as a Geometry object.
    """

    reactant_proton = structure_one.pop_atom(
        mol.proton_position_indices[attempt_number]
    )
    product_proton = structure_two.pop_atom(
        mol.proton_position_indices[attempt_number]
    )
    transition_proton = Atom("H", np.array([0., 0., (distance_multiplier - 0.01) / 2]))

    final_structure = structure_one + structure_two

    if stage == "Reactant":
        final_structure.add_atom(reactant_proton)
    elif stage == "Product":
        final_structure.add_atom(product_proton)
    elif stage == "Transition":
        final_structure.add_atom(transition_proton)

    return final_structure


def overlap_handler(
    structure_one: Geometry,
    structure_two: Geometry,
    stage: str,
    mol: Molecule,
    z_rotation_offset: npt.NDArray,
    distance_multiplier: float = 2.7,
) -> tuple[Geometry, Geometry, int]:
    """Fix molecule overlap issues

    First tries to rotate the molecule around the Z-axis,
    if that fails, switches to a different proton position.

    Parameters
    ----------
    structure_one : ArrayLike
        The structure generated from `structure_one_generator()`.
    structure_two : ArrayLike
        The structure generated from `structure_two_generator()`.
    stage : {"Reactant", "Product", "Transition"}
        Stage of the reaction..
    mol : Molecule
        The current Molecule object.
    z_rotation_offset : NDArray
        A matrix that rotates around the Z-axis.
    distance_multiplier : float, (default=2.7)
        The distance between the parent charge sites in Angstroms.

    Returns
    -------
    structure_one : list[list[float]]
        A corrected version of `structure_one`.
    structure_two : list[list[float]]
        A corrected version of `structure_two`.
    attempt_number : {0, 1, 2}
        Analogous to the proton position used.
    """

    proton_position_attempts = len(mol.proton_position_indices)
    attempt_number = 0

    while (
        structure_checker(structure_one, structure_two, mol, attempt_number)
        and attempt_number < proton_position_attempts
    ):
        z_rotation_offset = Z_ROTATION_180

        proton_position = mol.get_proton_position(attempt_number)

        alignment_matrix = LinearAlgebra.gen_alignment_matrix(proton_position, Z_UNIT_VECTOR)

        structure_one = structure_one_generator(
            alignment_matrix, mol
        )

        z_flipper = LinearAlgebra.gen_alignment_matrix(
            vector_one=structure_one.get_coords()[mol.get_proton_index(attempt_number)],
            vector_two=structure_one.get_coords()[mol.neighboring_atom_index],
            alignment_angle=np.pi
        )

        structure_two = structure_two_generator(
            stage,
            z_rotation_offset,
            structure_one,
            distance_multiplier,
            z_flipper,
            disable_nudge=True,
        )

        # If the structure works, then return it and exit the function
        if not structure_checker(
            structure_one, structure_two, mol, attempt_number
        ):
            print("Geometry was fixed with proton position #" + str(attempt_number))
            return structure_one, structure_two, attempt_number

        i = 0
        while (
            structure_checker(
                structure_one, structure_two, mol, attempt_number
            )
            and i < 8
        ):
            # 45 degree increments
            rotation_angle = i * (np.pi / 4)

            z_rotation_offset = np.array(
                [
                    [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
                    [np.sin(rotation_angle), np.cos(rotation_angle), 0],
                    [0, 0, 1],
                ],
                dtype=float,
            )

            z_flipper = LinearAlgebra.gen_alignment_matrix(
                vector_one=structure_one.get_coords()[mol.get_proton_index(attempt_number)],
                vector_two=structure_one.get_coords()[mol.neighboring_atom_index],
                alignment_angle=np.pi
            )

            structure_two = structure_two_generator(
                stage,
                z_rotation_offset,
                structure_one,
                distance_multiplier,
                z_flipper,
                disable_nudge=True,
            )

            if not structure_checker(
                structure_one, structure_two, mol, attempt_number
            ):
                print(
                    f"Geometry was fixed with proton position #{attempt_number} with a rotation of {i * 45} degrees."
                )
                return structure_one, structure_two, attempt_number

            i += 1

        attempt_number += 1
    print("Geometry could not be fixed.")
    return structure_one, structure_two, attempt_number


def open_folder(
    initial_dir: PathLike | str | None = None,
    must_exist: bool = False,
    title: str = "Select Directory"
):
    """Open a file browser dialog to select a folder.
    
    Parameters
    ----------
    initial_dir : PathLike | str | None, default=None
        The initial directory for the file dialog. Leaving this blank makes the file path the current working directory.
    must_exist : bool, default=False
        Require the directory to already exist.
    title : str, default="Select Directory"
        The title of the file dialog, should help the user understand what the dialog is for.

    Returns
    -------
    dir_path : PathLike
        The path to the directory that was selected.
    """

    if initial_dir is None:
        initial_dir = Path.cwd()

    root = Tk()
    root.wm_attributes("-topmost", 1)
    root.withdraw()

    dir_path = filedialog.askdirectory(
        initialdir=initial_dir,
        mustexist=must_exist,
        parent=root,
        title=title,
    )

    root.destroy()

    return Path(dir_path)


def open_file(
    default_extension: str | None = "",
    file_types: list[list[str] | str] | None = None,
    initial_dir: PathLike | None = None,
    initial_file: PathLike | None = None,
    title: str = "Select a file.",
):
    """Open a file browser dialog to select a file.

    Parameters
    ----------
    default_extension : str | None = ""
        The default extension that is appended to file names if they do not have an extension specified.
    file_types : list[str, str] | None = None
        The file types that are allowed to be selected. Format is `[[name1, ext1], [name2, ext2], ...]`.
    initial_dir : PathLike | str | None, default=None
        The initial directory for the file dialog. Leaving this blank makes the file path the current working directory.
    initial_file : PathLike | None, default=None
        An initial filename that is displayed in the dialog.
    title : str, default="Select Directory"
        The title of the file dialog, should help the user understand what the dialog is for.

    Returns
    -------
    file_path : PathLike
        The path to the file that was selected.
    """
    
    if initial_dir is None:
        initial_dir = Path.cwd()

    if file_types is not None:
        file_types = tuple((type[0], type[1]) for type in file_types)
    else:
        file_types = ()

    root = Tk()
    root.wm_attributes("-topmost", 1)
    root.withdraw()

    file_path = filedialog.askopenfilename(
        defaultextension=default_extension,
        filetypes=file_types,
        initialdir=initial_dir,
        initialfile=initial_file,
        parent=root,
        title=title,
    )

    root.destroy()

    return Path(file_path)


class Data:
    """Class to contain a list of SMILES and names for generating NEB or single-molecule structures.
    
    Attributes
    ----------
    smiles_strings : list[str]
        A list of the SMILES strings for the molecules.
    names : list[str]
        A list of the names of each molecule.

    Methods
    -------
    build_neb(index, stage, xyz_save_dir, xyz_data_dir, disable_nudge, distance_multiplier)
        Create a single-atom transfer NEB structure for a specified molecule in the list.
    build_single(index, xyz_save_dir)
        Create a single molecule structure from a SMILES string in the list.
    build_all_neb(stage, xyz_save_dir, xyz_data_dir, disable_nudge, distance_multiplier)
        Create single-atom transfer NEB structures for all molecules in the list.
    build_all_single(xyz_save_dir)
        Create a single molecule structure for all molecules in the list from their SMILES strings.
    multiple_input()
        Load a CSV containing molecule SMILES in the first column (with header) and molecule names in the second column (with header).

    Notes
    -----
    A CSV for this class should look like the following:

    ```
    smiles_header,names_header
    smiles1,name1
    smiles2,name2
    ...
    ```
    """

    def __init__(
        self,
        smiles_strings: list[str],
        names: list[str],
    ):
        self.smiles_strings: list[str] = smiles_strings
        self.names: list[str] = names


    def __repr__(self):
        self_repr = f"{"":6}{"Name":12}{"SMILES":11}\n\n"
        for i, smiles in enumerate(self.smiles_strings):
            self_repr += f"{str(i+1)+".":6}{self.names[i]:12}{smiles:20}\n"
        return self_repr


    def build_neb(
        self,
        index: int,
        stage: str = "Reactant",
        xyz_save_dir: PathLike | None = None,
        xyz_data_dir: PathLike | None = None,
        disable_nudge: bool = False,
        distance_multiplier: float = 2.7,
    ):
        """Build structures for single-atom transfer NEB calculations and write
        the result to an XYZ file.

        Parameters
        ----------
        index : int
            The index of the molecule that should be used in the structure build.
        stage : {"Reactant", "Product", "Transition"}
            Stage of the reaction..
        xyz_save_dir : PathLike | None, default=None
            Path to the folder that the XYZ file(s) should be written to.
            Default `None` prompts user to select a folder.
        xyz_data_dir : PathLike | None, default=None
            Path to a folder containing pre-existing structures that will be used in generating the NEB structures.
            Default `None` generates structures based off of the SMILES strings.
        disable_nudge : bool, default=False
            Disables the part of the structure build that applies a slight 
            nudge to the structure to avoid local minima in geometry optimizations
        distance_multiplier : float, default=2.7
            The distance between the protonated atoms in the final NEB structure (in Angstroms).

        Notes
        -----
        The `xyz_data_dir` should have a list of XYZ files with the same name as the supplied names (with a `.xyz` extension).

        The nudge referred to by `disable_nudge` is an applied rotation of +/- 15 degrees around the X-axis and +/- 20 degrees around
        the Y-axis used to avoid the molecules falling into a local minima during geometry optimizations.
        Sometimes this can cause structure overlap when no nudge would provide a clean structure, but generally has no problems.

        The parameter `distance_multiplier` is a set distance between the 2 protonated atoms BEFORE the nudge is applied.
        The default of 2.7 Angstroms is consistent with a large amount of optimized reactant and product geometries, however
        some systems may require a longer bond. It is not recommended to go below 2.5 Angstroms, as this can often cause atoms to
        overlap during structure building and make it harder for the overlap handler to fix the issue.
        """

        if xyz_save_dir is None:
            xyz_save_dir = open_folder(title="Select a directory to save XYZ Files")

        if xyz_data_dir is not None:
            structure = Geometry.from_xyz(
                xyz_data_dir/Path(f"{self.names[index]}.xyz"),
                charge=Molecule.calculate_charge(self.smiles_strings[index])
            )
        else:
            structure = Geometry.from_smiles(self.smiles_strings[index])

        if structure.charge == 0:
            raise RuntimeError("Molecule must be protonated for NEB structure generation!")

        attempt_number = 0

        mol = Molecule(
            self.smiles_strings[index],
            structure,
            self.names[index],
        )

        print(f"Working on {mol.molecule_name}...")

        z_rotation_offset = IDENTITY_MATRIX

        proton_position = mol.get_proton_position(attempt_number)

        alignment_matrix = LinearAlgebra.gen_alignment_matrix(proton_position, Z_UNIT_VECTOR)

        structure_one = structure_one_generator(
            alignment_matrix, mol
        )

        z_flipper = LinearAlgebra.gen_alignment_matrix(
            vector_one=structure_one.get_coords()[mol.get_proton_index(attempt_number)],
            vector_two=structure_one.get_coords()[mol.neighboring_atom_index],
            alignment_angle=np.pi
        )

        structure_two = structure_two_generator(
            stage,
            z_rotation_offset,
            structure_one,
            distance_multiplier,
            z_flipper,
            disable_nudge,
        )

        attempt_number = 0

        if structure_checker(structure_one, structure_two, mol, 0):
            print("Atomic Overlap Detected, attempting to fix...")
            structure_one, structure_two, attempt_number = overlap_handler(
                structure_one,
                structure_two,
                stage,
                mol,
                z_rotation_offset,
                distance_multiplier,
            )

        final_structure = final_structure_generator(
            structure_one,
            structure_two,
            stage,
            mol,
            attempt_number,
            distance_multiplier,
        )

        final_structure.to_xyz(
            name=f"{mol.molecule_name}-{stage[0]}",
            xyz_dir=xyz_save_dir
        )

        print(f"Molecule {mol.molecule_name} complete!\n")


    def build_single(self, index: int, xyz_save_dir: PathLike | None = None):
        """Build a single molecule structure from the list of SMILES.
        
        Parameters
        ----------
        index : int
            The index of the molecule that should be used in the structure build.
        xyz_save_dir : PathLike | None, default=None
            Path to the directory that the XYZ file(s) should be written to.
            Default `None` prompts user to select a directory.

        Notes
        -----
        The stage for the molecules are automatically selected to be either `"C"` for charged or `"N"` for neutral
        depending on the charge calculated from the SMILES. These are appended to the end of the file name to delineate
        between the protonated and neutral forms of the molecules.
        """

        print(f"Working on {self.names[index]}...")

        if xyz_save_dir is None:
            xyz_save_dir = open_folder(title="Select a directory to save XYZ Files")

        structure = Geometry.from_smiles(self.smiles_strings[index])

        if structure.charge == 0:
            stage = "N"
        elif structure.charge > 0:
            stage = "C"

        structure.to_xyz(
            name=f"{self.names[index]}-{stage}",
            xyz_dir=xyz_save_dir
        )

        print(f"Molecule {self.names[index]} complete!\n")


    def build_all_neb(
        self,
        stage: str = "Reactant",
        xyz_save_dir: PathLike | None = None,
        xyz_data_dir: PathLike | None = None,
        disable_nudge: bool = False,
        distance_multiplier: float = 2.7,
    ):
        """Build structures for single-atom transfer NEB calculations for every molecule 
        in the list and write the result to an XYZ file.

        All parameters are passed to `Data.build_neb()`.
        """

        if xyz_save_dir is None:
            xyz_save_dir = open_folder(title="Select a directory to save XYZ Files")

        for i in range(len(self.names)):
            Data.build_neb(
                self,
                index=i,
                stage=stage,
                xyz_save_dir=xyz_save_dir,
                xyz_data_dir=xyz_data_dir,
                disable_nudge=disable_nudge,
                distance_multiplier=distance_multiplier,
            )


    def build_all_single(self, xyz_save_dir: PathLike | None = None):
        """Build single-molecule structures for all SMILES in the list.

        All parameters are passed to `Data.build_single()`.
        """

        if xyz_save_dir is None:
            xyz_save_dir = open_folder(title="Select a directory to save XYZ Files")

        for i in range(len(self.names)):
            Data.build_single(self, index=i, xyz_save_dir=xyz_save_dir)


    @classmethod
    def multiple_input(
        cls,
        file_path: PathLike | None = None,
    ) -> Data:
        """Read CSV files with multiple molecules and/or specify folder containing XYZ files."""

        if file_path is None:
            file_path = open_file(title="Select a CSV", file_types = [["CSV Files", "*.csv"], ["All files", "*"]])

        multiple_molecule_data = []

        with open(file_path, newline="") as csvfile:
            data = csv.reader(csvfile)
            for row in data:
                multiple_molecule_data.append(row)

        smiles_strings = [i[0] for i in multiple_molecule_data]
        names = [i[1] for i in multiple_molecule_data]

        return Data(smiles_strings, names)
    

def main(
    build_type: str = "neb",
    use_default: bool = True,
    smiles: list[str] | str | None = None,
    name: list[str] | str | None = None,
):

    xyz_save_dir = input("Select a directory to save XYZ files (or hit <ENTER> to use a file browser): ")

    if smiles is not None and name is not None:
        data = Data(smiles_strings=smiles, names=name)
    elif smiles is None and name is None:
        data = Data()
    elif (smiles is None) ^ (name is None):
        raise ValueError("Please include both a list of SMILES and a list of names!")
    elif len(smiles) != len(name):
        raise ValueError("The list of SMILES strings must be the same as the list of names!")

    if build_type == "neb":

        stage = input('Enter a stage ("Reactant", "Product", "Transition"): ')

        if input("Use pre-existing XYZ data for NEB structures? (Y/N): ").casefold() == "Y".casefold():
            xyz_data_dir = input(
                "Enter the path to the folder containing the XYZ files "
                "(or hit <ENTER> to use a file browser): "
            )
            if not xyz_data_dir:
                open_folder(title="Select folder containing XYZ file(s).")
            elif path.isdir(xyz_data_dir):
                xyz_data_dir = Path(xyz_data_dir)
            else:
                raise ValueError("Must enter valid folder path!")
        else:
            xyz_data_dir = None

        if not use_default:
            disable_nudge = bool(input("Disable nudge (True or False, default=False): "))
            distance_multiplier = float(input("Set distance multiplier (default=2.7): "))
        else:
            disable_nudge = False
            distance_multiplier = 2.7

        if input("Build all structures? (Y/N): ") == "Y":
            data.build_all_neb(
                stage=stage,
                xyz_save_dir=xyz_save_dir,
                xyz_data_dir=xyz_data_dir,
                disable_nudge=disable_nudge,
                distance_multiplier=distance_multiplier,
            )
        else:
            index = input("Select molecule number: ")
            data.build_neb(
                index=index,
                stage=stage,
                xyz_save_dir=xyz_save_dir,
                xyz_data_dir=xyz_data_dir,
                disable_nudge=disable_nudge,
                distance_multiplier=distance_multiplier,
            )

    elif build_type == "single":
        if input("Build all structures? (Y/N): ") == "Y":
            data.build_all_single(xyz_save_dir=xyz_save_dir)
        else:
            index = input("Select molecule number: ")
            data.build_single(
                index=index,
                xyz_save_dir=xyz_save_dir,
            )



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "To start, enter either a list of SMILES strings and names "
            "or enter nothing to open a file browser and select a CSV with "
            "SMILES and names already specified.\n"
            "The format for a CSV with pre-existing SMILES and names should be\n"
            "'col_header1, col_header2'\n"
            "'smiles1, name1'\n"
            "'smiles2, name2'\n"
            "And so on for all molecules.\n\n"
            "Then, select a build type ('neb', 'single'), and optionally toggle defaults off if desired.\n"
            "The default values are for the distance between protonated atoms in NEB calculations (2.7 Angstrom) "
            "and for whether or not to apply a deviation to the NEB structures in order to avoid a local minima during "
            "geometry optimizations.\n\n"
            "Follow the prompts to either build all molecules in the list or build only one, "
            "and enter any information as requested."
        )
    )

    parser.add_argument('Build Type ("neb", "single"): ', help='Type of structure generation to run.')
    parser.add_argument()