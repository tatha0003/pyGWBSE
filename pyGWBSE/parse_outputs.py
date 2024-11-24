import json
import os
import re
from collections import defaultdict
from datetime import datetime

import numpy as np
from fireworks import FiretaskBase, FWAction, explicit_serialize
from fireworks.utilities.fw_serializers import DATETIME_HANDLER
from monty.json import MontyEncoder, jsanitize
from monty.os.path import zpath
from pydash.objects import get, has
from pymatgen.analysis.elasticity.elastic import ElasticTensor, ElasticTensorExpansion
from pymatgen.analysis.elasticity.strain import Deformation, Strain
from pymatgen.analysis.elasticity.stress import Stress
from pymatgen.analysis.ferroelectricity.polarization import (
    EnergyTrend,
    Polarization,
    get_total_ionic_dipole,
)
from pymatgen.analysis.magnetism import (
    CollinearMagneticStructureAnalyzer,
    Ordering,
    magnetic_deformation,
)
from pymatgen.command_line.bader_caller import bader_analysis_from_path
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.boltztrap import BoltztrapAnalyzer
from pymatgen.io.vasp.sets import get_vasprun_outcar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from atomate.common.firetasks.glue_tasks import get_calc_loc
from atomate.utils.utils import env_chk, get_logger, get_meta_from_structure
from atomate.vasp.config import DEFUSE_UNSUCCESSFUL, STORE_VOLUMETRIC_DATA
from atomate.vasp.database import VaspCalcDb
from atomate.vasp.drones import BADER_EXE_EXISTS, VaspDrone

__author__ = "Anubhav Jain, Kiran Mathew, Shyam Dwaraknath"
__email__ = "ajain@lbl.gov, kmathew@lbl.gov, shyamd@lbl.gov"

logger = get_logger(__name__)


@explicit_serialize
class BSEToDb(FiretaskBase):
    """
    Enter a BSE run into the database. 

    Optional params:
        db_file (str): path to file containing the database credentials.
            Supports env_chk. Default: write data to JSON file.
        gwcalc_dir (str): path to GW step of the calculation 
    """

    optional_params = [
        "gwcalc_dir", 
        "parse_dos",
        "bandstructure_mode",
        "additional_fields",
        "db_file",
        "fw_spec_field",
        "defuse_unsuccessful",
        "task_fields_to_push",
        "parse_chgcar",
        "parse_aeccar",
        "parse_potcar_file",
        "parse_bader",
        "store_volumetric_data",
    ]

    def run_task(self, fw_spec):

        calc_dir = get_calc_loc(self["gwcalc_dir"],
                                fw_spec["calc_locs"]) if self.get(
            "gwcalc_dir") else {}

        # parse the VASP directory
        logger.info(f"PARSING DIRECTORY: {calc_dir}")

        drone = VaspDrone(
            additional_fields=self.get("additional_fields"),
            parse_dos=self.get("parse_dos", False),
            parse_potcar_file=self.get("parse_potcar_file", True),
            bandstructure_mode=self.get("bandstructure_mode", False),
            parse_bader=self.get("parse_bader", BADER_EXE_EXISTS),
            parse_chgcar=self.get("parse_chgcar", False),  # deprecated
            parse_aeccar=self.get("parse_aeccar", False),  # deprecated
            store_volumetric_data=self.get(
                "store_volumetric_data", STORE_VOLUMETRIC_DATA
            ),
        )

        # assimilate (i.e., parse)
        d = drone.assimilate(calc_dir["path"])

        # add the structure
        bse_dir = os.getcwd()
        vrun, outcar = get_vasprun_outcar(bse_dir, parse_eigen=False, parse_dos=False)
        structure = vrun.final_structure
        d["optical_transition"]=vrun.optical_transition
        d["dielectric"]=vrun.dielectric

        db_file = env_chk(self.get("db_file"), fw_spec)
        d = jsanitize(d)

        if not db_file:
            del d["optical_transition"]
            with open(os.path.join(bse_dir, "BSE.json"), "w") as f:
                f.write(json.dumps(d, default=DATETIME_HANDLER))
        else:
            db = VaspCalcDb.from_db_file(db_file, admin=True)

            # optical transitions gets inserted into GridFS
            optical_transition = json.dumps(d["optical_transition"], cls=MontyEncoder)
            fsid, compression = db.insert_gridfs(
                optical_transition, collection="optical_transition_fs", compress=True
            )
            d["optical_transtiion_fs_id"] = fsid

            dielectric = json.dumps(d["dielectric"], cls=MontyEncoder)
            fsid, compression = db.insert_gridfs(
                dielectric, collection="dielectric_fs", compress=True
            )

            d["dielectric_fs_id"] = fsid
            del d["dielectric"]
            del d["optical_transition"]

            db.collection = db.db["BSE_results"]
            db.collection.insert_one(d)


