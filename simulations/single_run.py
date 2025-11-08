import matplotlib.pyplot as plt
from src.genetic_algorithm import ComponentGA

def main():
    # Run the genetic algorithm
    ga = ComponentGA()
    fitness_hist, cost_hist, fail_hist = ga.evolve()

    # --- Plot results ---
    plt.figure(figsize=(8, 5))
    plt.plot(fitness_hist, label="Best Fitness (Health)")
    plt.plot(cost_hist, label="Avg Cost")
    plt.plot(fail_hist, label="Avg Failures")
    plt.title("Maintenance Optimization with Failure Dynamics")
    plt.xlabel("Generation")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
