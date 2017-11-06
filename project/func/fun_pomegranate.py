from pomegranate import *
# The guests initial door selection is completely random
guest = DiscreteDistribution( { 'A': 1./3, 'B': 1./3, 'C': 1./3 } )

# The door the prize is behind is also completely random
prize = DiscreteDistribution( { 'A': 1./3, 'B': 1./3, 'C': 1./3 } )

# Monty is dependent on both the guest and the prize.
monty = ConditionalProbabilityTable(
    [[ 'A', 'A', 'A', 0.0 ],
     [ 'A', 'A', 'B', 0.5 ],
     [ 'A', 'A', 'C', 0.5 ],
     [ 'A', 'B', 'A', 0.0 ],
     [ 'A', 'B', 'B', 0.0 ],
     [ 'A', 'B', 'C', 1.0 ],
     [ 'A', 'C', 'A', 0.0 ],
     [ 'A', 'C', 'B', 1.0 ],
     [ 'A', 'C', 'C', 0.0 ],
     [ 'B', 'A', 'A', 0.0 ],
     [ 'B', 'A', 'B', 0.0 ],
     [ 'B', 'A', 'C', 1.0 ],
     [ 'B', 'B', 'A', 0.5 ],
     [ 'B', 'B', 'B', 0.0 ],
     [ 'B', 'B', 'C', 0.5 ],
     [ 'B', 'C', 'A', 1.0 ],
     [ 'B', 'C', 'B', 0.0 ],
     [ 'B', 'C', 'C', 0.0 ],
     [ 'C', 'A', 'A', 0.0 ],
     [ 'C', 'A', 'B', 1.0 ],
     [ 'C', 'A', 'C', 0.0 ],
     [ 'C', 'B', 'A', 1.0 ],
     [ 'C', 'B', 'B', 0.0 ],
     [ 'C', 'B', 'C', 0.0 ],
     [ 'C', 'C', 'A', 0.5 ],
     [ 'C', 'C', 'B', 0.5 ],
     [ 'C', 'C', 'C', 0.0 ]], [guest, prize])

# State objects hold both the distribution, and a high level name.
s1 = State(guest, name="guest")
s2 = State(prize, name="prize")
s3 = State(monty, name="monty")

# Create the Bayesian network object with a useful name
model = BayesianNetwork("Monty Hall Problem")

# Add the three states to the network
model.add_states(s1, s2, s3)

# Add transitions which represent conditional dependencies, where the second node is conditionally dependent on the first node (Monty is dependent on both guest and prize)
model.add_transition(s1, s3)
model.add_transition(s2, s3)
model.bake()


print(model.probability(['A', 'B', 'C']))
print(model.probability(['B', 'B', 'B']))
print(model.log_probability(['C', 'A', 'B']))
print(model.log_probability(['B', 'A', 'A']))


print(model.predict_proba({}))


marginals = model.predict_proba({})
print(model.predict_proba({}))


model.predict_proba({'guest': 'A'})
model.predict_proba({'guest': 'A', 'monty': 'C'})

model.predict([['B', 'A', None],
               ['C', 'A', None],
               ['B', 'C', None],
               ['A', 'B', None]])

model.fit([['A', 'B', 'C'],
           ['A', 'C', 'B'],
           ['A', 'A', 'C'],
           ['B', 'B', 'C'],
           ['B', 'C', 'A']])

print(model.predict_proba({}))



import time
import numpy
times = []
for i in range(2, 18):
    tic = time.time()
    X = numpy.random.randint(2, size=(100, i))
    model = BayesianNetwork.from_samples(X, algorithm='exact')
    times.append(time.time() - tic)

import seaborn
seaborn.set_style('whitegrid')

import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))
plt.title('Time To Learn Bayesian Network', fontsize=18)
plt.xlabel("Number of Variables", fontsize=14)
plt.ylabel("Time (s)", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.plot(range(2, 18), times, linewidth=3, color='c')
plt.yscale('log')
plt.show()



times = []
for i in range(2, 253, 10):
    tic = time.time()
    X = numpy.random.randint(2, size=(10000, i))
    model = BayesianNetwork.from_samples(X, algorithm='chow-liu')
    times.append( time.time() - tic )


import seaborn
seaborn.set_style('whitegrid')

plt.figure(figsize=(14, 6))
plt.title('Time To Learn Bayesian Network', fontsize=18)
plt.xlabel("Number of Variables", fontsize=14)
plt.ylabel("Time (s)", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.plot( range(2, 253, 10), times, linewidth=3, color='c')
plt.yscale('log')