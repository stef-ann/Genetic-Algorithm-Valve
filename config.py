# --- Tank system parameters ---
INITIAL_WATER_LEVEL = 25
TIME_STEPS = 100
LAMBDA_FACTOR = 0.7  # weight for reliability gain vs cost

# --- Genetic Algorithm parameters ---
POPULATION_SIZE = 10
NUM_GENERATIONS = 1000
MUTATION_RATE = 0.347
CROSSOVER_RATE = 0.7

# --- Random seed for reproducibility ---
SEED = 42

# --- Weibull Reliability Model Parameters ---
# β (shape): <1 early-life, =1 random, >1 wear-out
BETA = 0.7    # try 0.7 (early), 1.0 (random), 2.5 (wear-out)
# η (scale): characteristic life (larger η = longer component lifespan)
ETA = 50
