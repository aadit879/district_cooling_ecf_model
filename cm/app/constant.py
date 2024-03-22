
CELERY_BROKER_URL_DOCKER = 'amqp://admin:mypass@rabbit:5672/'
CELERY_BROKER_URL_LOCAL = 'amqp://localhost/'


CM_REGISTER_Q = 'rpc_queue_CM_register' # Do no change this value

CM_NAME = 'CM - District Cooling'
RPC_CM_ALIVE= 'rpc_queue_CM_ALIVE' # Do no change this value
RPC_Q = 'rpc_queue_CM_compute' # Do no change this value
CM_ID = 20 # CM_ID is defined by the enegy research center of Martigny (CREM)
PORT_LOCAL = int('500' + str(CM_ID))
PORT_DOCKER = 80

#TODO ********************setup this URL depending on which version you are running***************************

CELERY_BROKER_URL = CELERY_BROKER_URL_DOCKER
PORT = PORT_DOCKER

#TODO ********************setup this URL depending on which version you are running***************************

TRANFER_PROTOCOLE ='http://'
INPUTS_CALCULATION_MODULE = [
    {'input_name': 'Average Electricity Price',
     'input_type': 'input',
     'input_parameter_name': 'electricity_prices',
     'input_value': '30',
     'input_priority': 0,
     'input_unit': 'Euro/MWh',
     'input_min': 0,
     'input_max': 1000, 'cm_id': CM_ID  # Do no change this value
     },

    {'input_name': 'Estimated Cooling Days in a year',
     'input_type': 'input',
     'input_parameter_name': 'flh_cooling_days',
     'input_value': '60',
     'input_priority': 0,
     'input_unit': 'Days',
     'input_min': 0,
     'input_max': 300, 'cm_id': CM_ID  # Do no change this value
     },

    {'input_name': 'COP District Cooling Supply Technology',
     'input_type': 'input',
     'input_parameter_name': 'COP_DC',
     'input_value': '4.89',
     'input_priority': 0,
     'input_unit': '',
     'input_min': 0,
     'input_max': 100, 'cm_id': CM_ID  # Do no change this value
     },

    {'input_name': 'Network Operating Conditions',
     'input_type': 'input',
     'input_parameter_name': 'delta_T_dc',
     'input_value': '10',
     'input_priority': 0,
     'input_unit': '°C',
     'input_min': 10,
     'input_max': 100, 'cm_id': CM_ID  # Do no change this value
     },

    {'input_name': 'COP Individual Cooling Supply Technology',
     'input_type': 'input',
     'input_parameter_name': 'ind_tec_SEER',
     'input_value': '3.6',
     'input_priority': 0,
     'input_unit': '',
     'input_min': 0,
     'input_max': 100, 'cm_id': CM_ID  # Do no change this value
     },

    {'input_name': 'Interest Rate',
     'input_type': 'input',
     'input_parameter_name': 'interest_rate',
     'input_value': '6',
     'input_priority': 1,
     'input_unit': '%',
     'input_min': 0,
     'input_max': 1, 'cm_id': CM_ID  # Do no change this value
     },

    {'input_name': 'Depreciation Time District Cooling',
     'input_type': 'input',
     'input_parameter_name': 'depreciation_dc',
     'input_value': '25',
     'input_priority': 1,
     'input_unit': 'years',
     'input_min': 0,
     'input_max': 100, 'cm_id': CM_ID  # Do no change this value
     },

     {'input_name': 'Depreciation Time District Cooling AC',
     'input_type': 'input',
     'input_parameter_name': 'depreciation_ac',
     'input_value': '15',
     'input_priority': 1,
     'input_unit': 'years',
     'input_min': 0,
     'input_max': 100, 'cm_id': CM_ID  # Do no change this value
     },

    {'input_name': 'Depreciation Time Individual Cooling',
     'input_type': 'input',
     'input_parameter_name': 'depreciation_ind',
     'input_value': '15',
     'input_priority': 1,
     'input_unit': 'years',
     'input_min': 0,
     'input_max': 100, 'cm_id': CM_ID  # Do no change this value
     },
]


SIGNATURE = {

     "category": "Demand",
    "authorized_scale":["LAU 2","Hectare"],
    "cm_name": CM_NAME,
    "wiki_url": "https://wiki.hotmaps.hevs.ch/en/CM-Scale-heat-and-cool-density-maps", #TODO AM: Needs to be updated
    "layers_needed": ["cool_tot_curr_density_tif","gfa_tot_curr_density_tif","gfa_nonres_curr_density_tif"], # AM: nonres to be selected by default (for now)
    "type_layer_needed": [
        {"type": "heat", "description": "Select the cold demand density layer"},
        {"type": "gross_floor_area", "description": "Select the gross floor area density layer"},
        {"type": "gfa_nonres_curr_density", "description": "Select the gross floor area density layer non-residential"}
    ],
    "type_vectors_needed": [], ### AM: to be added; most likely convert the vector files to raster
    "cm_url": "Do not add something",
    "cm_description": "This calculation module allows the identification of areas with district cooling potential",
    "cm_id": CM_ID,
    'inputs_calculation_module': INPUTS_CALCULATION_MODULE
}
