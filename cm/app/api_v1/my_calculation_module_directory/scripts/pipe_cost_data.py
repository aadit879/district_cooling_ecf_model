
#from scripts import read_inputs

# in case the pipe costs re updated
# def update_pipe_cost_data():
#     pipe_costs = pd.read_csv(read_inputs.input_directory + 'Pipe_costs.csv', encoding='unicode_escape')
#     # update data
#     return pipe_costs


## Pipe cost data calculated based on the Pearsonn heating  adjusted for cooling pipe sizes
# costs include both material and trench digging costs
# no pipe description; (insluated)
data = {
    'Size_DN': [20, 25, 32, 40, 50, 65, 80, 100, 125, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500],
    'Cost_Eur/m': [433.260, 454.325, 483.816, 517.520, 559.650, 622.845, 686.040, 770.300, 875.625, 980.950, 1191.600, 1402.250, 1612.900, 1823.550, 2034.200, 2244.850, 2455.500, 2876.800, 3298.100, 3719.400, 4140.700, 4562.000, 4983.300, 5404.600, 5825.900, 6247.200, 6668.500]
}


