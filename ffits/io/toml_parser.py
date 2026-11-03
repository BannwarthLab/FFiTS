"""Applies command-line overrides on top of a loaded CalculationData object."""

import logging
from ffits.datatype.calculation_data import CalculationData

logger = logging.getLogger(__name__)


def overwrite_from_commandline(
    calcdata: CalculationData,
    multiplicity: int | None,
    charge: int | None,
    structures: list | None = None,
    config_file: str | None = None,
):
    """overwrites information in CalculationData object if given via commandline

    Warns if the command line changes a charge/multiplicity set by ``config_file``.
    """
    if multiplicity is not None:
        if config_file is not None and multiplicity != calcdata.system.multiplicity:
            logger.warning(
                f"Multiplicity {multiplicity} from the command line overrides multiplicity {calcdata.system.multiplicity} from {config_file}."
            )
        calcdata.system.multiplicity = multiplicity
    if charge is not None:
        if config_file is not None and charge != calcdata.system.charge:
            logger.warning(
                f"Charge {charge} from the command line overrides charge {calcdata.system.charge} from {config_file}."
            )
        calcdata.system.charge = charge
    if structures:
        if len(structures) >= 1:
            calcdata.reactant_path.xyz_filename = structures[0]
            logger.info(f"Reactant XYZ file set to: {structures[0]}")
        if len(structures) >= 2:
            calcdata.product_path.xyz_filename = structures[1]
            logger.info(f"Product XYZ file set to: {structures[1]}")
        if len(structures) > 2:
            logger.warning(
                "More than 2 structure files provided. Only the first two will be used."
            )
