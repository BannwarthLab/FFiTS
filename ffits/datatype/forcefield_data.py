#!/bin/python
"""The ForceField class: parameters, reference values and I/O for a force field."""

import numpy as np
import pandas as pd
import re
from collections.abc import Callable
from ffits.utils.geometry import angstrom2bohr, convert_xyz_to_fortranstyle
import logging

logger = logging.getLogger(__name__)


class ForceField:
    """
    FF definition through FF parameters (np arrays starting with ``c_``), reference values (bondlenghts, angles, etc) and corresponding atom numbers, which construct the bond / angle / dihedral angle / lj term.
    """

    def __init__(
        self,
        nat: int,
        ff_filename: str,
        readff: bool = False,
        energy_calculator: Callable = None,
        gradient_calculator: Callable = None,
        hessian_calculator: Callable = None,
    ):
        """Initialize the force field, optionally reading parameters from a file.

        Args:
            nat (int): Number of atoms the force field applies to.
            ff_filename (str): Path to the force field csv file.
            readff (bool, optional): If True, read parameters from
                ``ff_filename`` immediately. Defaults to False.
            energy_calculator (Callable, optional): Function used by :meth:`get_energy`.
            gradient_calculator (Callable, optional): Function used by :meth:`get_gradient`.
            hessian_calculator (Callable, optional): Function used by :meth:`get_hessian`.
        """
        self.nat = nat
        self.ff_filename = ff_filename
        self.columns = ["type", "atoms", "parameter", "reference_value"]
        self.bonds: pd.DataFrame = pd.DataFrame(columns=self.columns)
        self.angles: pd.DataFrame = pd.DataFrame(columns=self.columns)
        self.dihedrals: pd.DataFrame = pd.DataFrame(columns=self.columns)
        self.repulsive: pd.DataFrame = pd.DataFrame(columns=self.columns)
        if readff:
            self.readin(ff_filename)
        self.energy_calculator = energy_calculator
        self.gradient_calculator = gradient_calculator
        self.hessian_calculator = hessian_calculator
        self.start_from_reactant: bool = True  # only relevant for TS FF

    def get_energy(self, xyz_displaced: np.ndarray) -> float:
        """
        Calculates energy with defined self.energy_calculator. Handles transfer in correct xyz format for said calculator.

        Args:
            xyz_displaced (np.ndarray): Displaced xyz. Can be in shape (nat, 3) or (nat*3,).

        Returns:
            energy (float): The calculated energy.
        """
        if np.shape(xyz_displaced) == (self.nat, 3):
            return self.energy_calculator(
                angstrom2bohr(convert_xyz_to_fortranstyle(xyz_displaced)),
                self,
            )
        if len(xyz_displaced) == self.nat * 3:
            return self.energy_calculator(
                angstrom2bohr(xyz_displaced.reshape(self.nat, 3).T), self
            )
        return self.energy_calculator(xyz_displaced, self)

    def get_gradient(self, xyz_displaced: np.ndarray) -> np.ndarray:
        """
        Calculates gradient with defined self.gradient_calculator. Handles transfer in correct xyz format for said calculator.

        Args:
            xyz_displaced (np.ndarray): Displaced xyz. Can be in shape (nat, 3) or (nat*3).

        Returns:
            gradient (np.ndarray, shape=(nat*3)): The calculated gradient.
        """
        if np.shape(xyz_displaced) == (self.nat, 3):
            return self.gradient_calculator(
                angstrom2bohr(convert_xyz_to_fortranstyle(xyz_displaced)),
                self,
            )
        if len(xyz_displaced) == self.nat * 3:
            return self.gradient_calculator(
                angstrom2bohr(xyz_displaced.reshape(self.nat, 3).T), self
            )
        return self.gradient_calculator(xyz_displaced, self)

    def get_hessian(self, xyz_displaced: np.ndarray) -> np.ndarray:
        """
        Calculates hessian with defined self.hessian_calculator. Handles transfer in correct xyz format for said calculator.

        Args:
            xyz_displaced (np.ndarray): Displaced xyz. Can be in shape (nat, 3) or (nat*3).

        Returns:
            hessian (np.ndarray, shape=(nat*3, nat*3)): The calculated hessian.
        """
        if np.shape(xyz_displaced) == (self.nat, 3):
            return self.hessian_calculator(
                angstrom2bohr(convert_xyz_to_fortranstyle(xyz_displaced)),
                self,
            )
        if len(xyz_displaced) == self.nat * 3:
            return self.hessian_calculator(
                angstrom2bohr(xyz_displaced.reshape(self.nat, 3).T), self
            )
        return self.hessian_calculator(xyz_displaced, self)

    def write(self):
        """
        Writes out all FF data in self.ff_filename in the csv format.
        """

        def format_atoms(t):
            """Format an atom-index tuple as a bracketed, space-separated string for the csv."""
            return "[" + " ".join(map(str, t)) + "]"

        df_combined = pd.concat(
            [self.bonds, self.angles, self.dihedrals, self.repulsive], ignore_index=True
        )
        df_combined = df_combined.copy()
        df_combined["atoms"] = df_combined["atoms"].apply(format_atoms)
        df_combined.to_csv(self.ff_filename)
        logger.info(f"FF information written to {self.ff_filename}.")

    def readin(self, filename: str):
        """
        Reads a force field csv file and fills defines the ForceField object accordingly.

        Args:
            filename (str): path to the force field csv file. The file should have columns atoms, parameter, reference_value and type, where type can be bonds, angles, dihedrals or repulsive. The atoms column should contain a tuple of 0-based atom indices, which are involved in the corresponding term. For example, for a bond between atom 0 and 1, the atoms column should contain (0,1).
        """
        try:

            def parse_atoms(x):
                """Parse 'atoms' column into a tuple of integers."""
                if pd.isna(x):
                    return tuple()
                if isinstance(x, (list, tuple)):
                    # Already iterable — ensure tuple of ints
                    return tuple(int(i) for i in x)
                # Remove brackets and commas, split on whitespace
                x = re.sub(r"[\[\],]", " ", str(x))
                tokens = x.split()
                return tuple(int(tok) for tok in tokens)

            df = pd.read_csv(
                filename,
                converters={
                    "atoms": parse_atoms,
                    "parameter": float,
                    "reference_value": float,
                },
                index_col=0,
            )

        except FileNotFoundError:
            raise FileNotFoundError(f"Force field file '{filename}' not found.")
        except pd.errors.EmptyDataError:
            raise ValueError(f"Force field file '{filename}' is empty or malformed.")
        except Exception as e:
            raise ValueError(f"Error while reading '{filename}': {e}")

        self.bonds = df[df["type"] == "bonds"].copy()
        self.angles = df[df["type"] == "angles"].copy()
        self.dihedrals = df[df["type"] == "dihedrals"].copy()
        self.repulsive = df[df["type"] == "repulsive"].copy()
