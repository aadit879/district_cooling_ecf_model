import unittest
from werkzeug.exceptions import NotFound
from app import create_app
import os.path
from shutil import copyfile
from .test_client import TestClient
UPLOAD_DIRECTORY = '/var/hotmaps/cm_files_uploaded'

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)
    os.chmod(UPLOAD_DIRECTORY, 0o777)


class TestAPI(unittest.TestCase):


    def setUp(self):
        self.app = create_app(os.environ.get('FLASK_CONFIG', 'development'))
        self.ctx = self.app.app_context()
        self.ctx.push()

        self.client = TestClient(self.app,)

    def tearDown(self):

        self.ctx.pop()


    def test_compute(self):
        # simulate copy from HTAPI to CM

        int_raster_gfa_tot = './tests/data/gfa_tot_curr_density_lau2.tif'
        int_raster_gfa_non_res = './tests/data/gfa_nonres_curr_density_lau2.tif'
        int_raster_cdm = './tests/data/Vienna_cool_tot_curr_density_lau2.tif'
        save_path1 = UPLOAD_DIRECTORY + '/gfa_tot_curr_density_lau2.tif'
        save_path2 = UPLOAD_DIRECTORY + "/gfa_nonres_curr_density_lau2.tif"
        save_path3 = UPLOAD_DIRECTORY + "/Vienna_cool_tot_curr_density_lau2.tif"


        copyfile(int_raster_gfa_tot, save_path1)
        copyfile(int_raster_gfa_non_res, save_path2)
        copyfile(int_raster_cdm, save_path3)

        inputs_raster_selection = {}
        inputs_parameter_selection = {}
        inputs_raster_selection["gross_floor_area"]  = save_path1
        inputs_raster_selection["gfa_nonres_curr_density"] = save_path2
        inputs_raster_selection["cooling"] = save_path3 # TODO: AM does this category changes to cold?


        inputs_parameter_selection['electricity_prices'] = "80"
        inputs_parameter_selection['flh_cooling_days'] = "60"
        inputs_parameter_selection['COP_DC'] = "4.89"
        inputs_parameter_selection['delta_T_dc'] = "10"
        inputs_parameter_selection['ind_tec_SEER'] = "3.6"
        inputs_parameter_selection['interest_rate'] = "0.06"
        inputs_parameter_selection['depreciation_dc'] = "25"
        inputs_parameter_selection['depreciation_ac'] = "15"

        # register the calculation module a
        payload = {"inputs_raster_selection": inputs_raster_selection,
                   "inputs_parameter_selection": inputs_parameter_selection}


        rv, json = self.client.post('computation-module/compute/', data=payload)

        self.assertTrue(rv.status_code == 200)


