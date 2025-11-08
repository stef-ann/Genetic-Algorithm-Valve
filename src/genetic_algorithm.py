import numpy as np
from src.tank_model import simulate_component
from config import POPULATION_SIZE, TIME_STEPS, NUM_GENERATIONS, MUTATION_RATE
from config import BETA, ETA


class ComponentGA:
    def __init__(self):
        self.population = np.random.rand(POPULATION_SIZE, TIME_STEPS, 2)
        self.best_history = []

    def evolve(self):
        self.best_history = []
        cost_history = []
        failure_history = []

        for gen in range(NUM_GENERATIONS):
            fitness, costs, failures = [], [], []

            # --- Run simulation for each chromosome ---
            for ind in self.population:
                result = simulate_component(ind)

                # Some versions of simulate_component may return (fitness, cost, failures)
                if isinstance(result, tuple):
                    f, c, fl = result
                else:
                    f, c, fl = result, 0, 0

                fitness.append(f)
                costs.append(c)
                failures.append(fl)

            # --- Collect generation stats ---
            fitness = np.array(fitness)
            best = np.max(fitness)
            self.best_history.append(best)
            cost_history.append(np.mean(costs))
            failure_history.append(np.mean(failures))

            # --- Selection + reproduction (same as before) ---
            survivors = self.population[np.argsort(fitness)[::-1][:POPULATION_SIZE // 2]]
            children = self._crossover(survivors)
            children = self._mutate(children)
            self.population = np.vstack((survivors, children))

            print(f"Gen {gen+1}/{NUM_GENERATIONS} | β={BETA:.2f}, η={ETA} | "
                f"Best Fitness: {best:.2f} | Avg Cost: {np.mean(costs):.2f} | Avg Failures: {np.mean(failures):.2f}")


        # Return multiple histories so you can plot them later
        return self.best_history, cost_history, failure_history


    def _crossover(self, parents):
        offspring = []
        for _ in range(len(parents)):
            p1, p2 = parents[np.random.randint(len(parents), size=2)]
            cut = np.random.randint(1, TIME_STEPS - 1)
            child = np.vstack((p1[:cut], p2[cut:]))
            offspring.append(child)
        return np.array(offspring)

    def _mutate(self, offspring):
        mask = np.random.rand(*offspring.shape) < MUTATION_RATE
        offspring[mask] += np.random.randn(*offspring[mask].shape) * 0.1
        np.clip(offspring, 0, 1, out=offspring)
        return offspring
