# Import necessary libraries
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

#  Define the structure of the Bayesian Network
model = BayesianNetwork([('Cloudy', 'Rain'), ('Cloudy', 'Temperature')])

# Define Conditional Probability Distributions (CPDs)
# P(Cloudy)
cpd_cloudy = TabularCPD(variable='Cloudy', variable_card=2, values=[[0.5], [0.5]])

# P(Rain | Cloudy)
cpd_rain = TabularCPD(variable='Rain', variable_card=2, 
                      values=[[0.8, 0.2], [0.2, 0.8]],
                      evidence=['Cloudy'], evidence_card=[2])

# P(Temperature | Cloudy)
cpd_temperature = TabularCPD(variable='Temperature', variable_card=3, 
                             values=[[0.3, 0.1],  # Cold
                                     [0.5, 0.4],  # Warm
                                     [0.2, 0.5]], # Hot
                             evidence=['Cloudy'], evidence_card=[2])

# Add CPDs to the model
model.add_cpds(cpd_cloudy, cpd_rain, cpd_temperature)

# Validate the model to check for errors
assert model.check_model()

# Perform Inference
inference = VariableElimination(model)

# Query the probability of Rain given that it is Cloudy and Warm
query_result = inference.query(variables=['Rain'], 
                               evidence={'Cloudy': 1, 'Temperature': 1})  # Cloudy = True, Temperature = Warm

# Display the result
print(query_result)