import random
import torch
import numpy as np

from digital_CNN import quantization_error_for_bins, create_analog_network, initialize_bins, extract_weights_from_model

class GeneticAlgorithm:
    def __init__(self, weights, pop_size=50, mutation_rate=0.1, generations=100, elite_size=10):
        self.weights = weights
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.elite_size = elite_size

    def initialize_population(self, pos_bins, neg_bins):
        population = []
        for _ in range(self.pop_size):
            random.shuffle(pos_bins)
            random.shuffle(neg_bins)
            population.append((pos_bins.copy(), neg_bins.copy()))
        return population
    
    def fitness(self, individual):
        pos_bins, neg_bins = individual
        error = quantization_error_for_bins(self.weights, pos_bins, neg_bins)
        return 1 / (1 + error)
    
    def select_parents(self, population):
        fitnesses = [self.fitness(ind) for ind in population]
        total_fitness = sum(fitnesses)
        probabilities = [f / total_fitness for f in fitnesses]
        parent_indices = np.random.choice(len(population), size=self.pop_size, p=probabilities, replace=True)
        parents = [population[i] for i in parent_indices]
        return parents
    
    def crossover(self, parent1, parent2):
        # One-point crossover for positive bins
        crossover_point = random.randint(1, len(parent1[0]) - 1)
        child1_pos = parent1[0][:crossover_point] + parent2[0][crossover_point:]
        child2_pos = parent2[0][:crossover_point] + parent1[0][crossover_point:]
        
        # One-point crossover for negative bins
        crossover_point = random.randint(1, len(parent1[1]) - 1)
        child1_neg = parent1[1][:crossover_point] + parent2[1][crossover_point:]
        child2_neg = parent2[1][:crossover_point] + parent1[1][crossover_point:]
        
        return (child1_pos, child1_neg), (child2_pos, child2_neg)
    
    def mutate(self, individual):
        pos_bins, neg_bins = individual
        for i in range(len(pos_bins)):
            if random.random() < self.mutation_rate:
                pos_bins[i] += random.choice([-1, 1]) * pos_bins[0]
        for i in range(len(neg_bins)):
            if random.random() < self.mutation_rate:
                neg_bins[i] += random.choice([-1, 1]) * neg_bins[0]
        return pos_bins, neg_bins
    
    def run(self):
        initial_pos_bins, initial_neg_bins = initialize_bins(self.weights, 10)
        population = self.initialize_population(initial_pos_bins, initial_neg_bins)
        
        for generation in range(self.generations):
            parents = self.select_parents(population)
            
            # Crossover and mutation to produce the next generation
            next_gen = []
            for i in range(0, len(parents), 2):
                child1, child2 = self.crossover(parents[i], parents[i + 1])
                next_gen.append(self.mutate(child1))
                next_gen.append(self.mutate(child2))
            
            # Keep the elite solutions
            population_sorted = sorted(population, key=lambda x: -self.fitness(x))
            next_gen_sorted = sorted(next_gen, key=lambda x: -self.fitness(x))
            next_gen = next_gen_sorted[:-self.elite_size] + population_sorted[:self.elite_size]
            
            population = next_gen
        
        # Return the best solution from the last generation
        best_solution = max(population, key=self.fitness)
        return best_solution

# Create an instance of the GA and run it

model = create_analog_network()
model.load_state_dict(torch.load('model_checkpoint.pth', map_location="cuda"))
model_weights = extract_weights_from_model(model)
ga = GeneticAlgorithm(model_weights)
best_pos_bins, best_neg_bins = ga.run()
print(best_pos_bins, best_neg_bins)