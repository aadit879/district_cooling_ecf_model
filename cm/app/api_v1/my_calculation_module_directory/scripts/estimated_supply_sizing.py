import geopandas as gpd


def supply_sizing(COP,fld, af,cf,r,T, output_polygon_name, output_directory, electircity_price_EurpKWh):
    vector = gpd.read_file(output_polygon_name)
    demand = vector.Tot_dem

    flh = fld * 24 * af * cf

    peak_load = demand / flh

    absorption_chiller_spec_inv_per_MW = 95800  # efficient heating cooling pathways
    absorption_chiller_fxd_per_MW = 16000  # https://doi.org/10.1016/j.apenergy.2020.116166

    total_investment_costs_Eur = peak_load * absorption_chiller_spec_inv_per_MW + peak_load * absorption_chiller_fxd_per_MW

    crf = (r * (1 + r) ** T) / (((1 + r) ** T) - 1)
    investment_anualized_EUR = total_investment_costs_Eur * crf

    # TODO: Check the assumption
    electrical_input = demand / COP  # assumption based on literature; electrical input is 5-15% of the total cooling demand
    variable_cost = electircity_price_EurpKWh * 1000
    annual_variable_costs = variable_cost * electrical_input

    total_system_costs_annualized_euros = investment_anualized_EUR + annual_variable_costs
    total_investment_annualized = vector.Inv_gp + total_system_costs_annualized_euros

    LCOC_DC = total_investment_annualized / vector.Tot_dem

    vector.loc[:, 'TOTAL_INV'] = total_investment_annualized.values  # total system + network costs of the DC grid
    vector.loc[:, 'LCOC_DC'] = LCOC_DC.values

    # vector = vector[vector['LCOC_fin']<vector['Avg_LCOCin']]

    vector.to_file(output_polygon_name)

    return vector


def supply_resizing(demand, electircity_price_EurpKWh):
    COP = 4.89
    heat_source_energy = demand / COP

    fld = 60  # aligning with the assumptions of the calculation_module_main
    af = 0.9
    cf = 0.5
    flh = fld * 24 * af * cf

    peak_load = demand / flh

    absorption_chiller_spec_inv_per_MW = 95800  # effecient heating cooling pathays
    absorption_chiller_fxd_per_MW = 16000  # https://doi.org/10.1016/j.apenergy.2020.116166

    total_investment_costs_Eur = peak_load * absorption_chiller_spec_inv_per_MW + peak_load * absorption_chiller_fxd_per_MW

    r = 0.06
    T = 30
    crf = (r * (1 + r) ** T) / (((1 + r) ** T) - 1)
    investment_anualized_EUR = total_investment_costs_Eur * crf

    #electrical_input = demand * 0.15  # assumption based on literature; electrical input is 5-15% of the total cooling demand
    electrical_input = demand / COP
    variable_cost = electircity_price_EurpKWh * 1000
    annual_variable_costs = variable_cost * electrical_input

    total_system_costs_annualized_euros = investment_anualized_EUR + annual_variable_costs

    return peak_load, total_system_costs_annualized_euros



