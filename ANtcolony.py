import numpy as np

class AntColonyOptimization:
    def __init__(self, num_ants, num_iterations, alpha, beta, rho, Q):
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha  # pheromone factor
        self.beta = beta  # heuristic factor
        self.rho = rho  # evaporation rate
        self.Q = Q  # pheromone deposit factor

    def solve(self, distance_matrix):
        num_cities = distance_matrix.shape[0]
        pheromone_matrix = np.ones((num_cities, num_cities))

        best_path = None
        best_distance = np.inf

        for _ in range(self.num_iterations):
            ant_paths = self.construct_ant_paths(pheromone_matrix, distance_matrix)
            self.update_pheromone(pheromone_matrix, ant_paths)

            shortest_path_index = np.argmin([self.calculate_path_distance(path, distance_matrix) for path in ant_paths])
            shortest_path = ant_paths[shortest_path_index]
            shortest_distance = self.calculate_path_distance(shortest_path, distance_matrix)

            if shortest_distance < best_distance:
                best_path = shortest_path
                best_distance = shortest_distance

            pheromone_matrix *= self.rho

        return best_path, best_distance

    def construct_ant_paths(self, pheromone_matrix, distance_matrix):
        num_cities = pheromone_matrix.shape[0]
        ant_paths = []

        for ant in range(self.num_ants):
            start_city = np.random.randint(num_cities)
            visited_cities = [start_city]
            path = [start_city]

            while len(visited_cities) < num_cities:
                next_city = self.select_next_city(pheromone_matrix, distance_matrix, visited_cities)
                visited_cities.append(next_city)
                path.append(next_city)

            ant_paths.append(path)

        return ant_paths

    def select_next_city(self, pheromone_matrix, distance_matrix, visited_cities):
        current_city = visited_cities[-1]
        num_cities = pheromone_matrix.shape[0]

        unvisited_cities = list(set(range(num_cities)) - set(visited_cities))
        probabilities = np.zeros(num_cities)

        for city in unvisited_cities:
            pheromone = pheromone_matrix[current_city, city]
            distance = distance_matrix[current_city, city]
            probabilities[city] = pheromone ** self.alpha * (1.0 / distance) ** self.beta

        probabilities /= np.sum(probabilities)
        next_city = np.random.choice(range(num_cities), p=probabilities)

        return next_city

    def update_pheromone(self, pheromone_matrix, ant_paths):
        num_cities = pheromone_matrix.shape[0]

        for path in ant_paths:
            path_distance = self.calculate_path_distance(path, distance_matrix)

            for i in range(num_cities - 1):
                current_city = path[i]
                next_city = path[i + 1]
                pheromone_matrix[current_city, next_city] += self.Q / path_distance

    def calculate_path_distance(self, path, distance_matrix):
        distance = 0

        for i in range(len(path) - 1):
            current_city = path[i]
            next_city = path[i + 1]
            distance += distance_matrix[current_city, next_city]

        return distance
# Create a distance matrix representing the distances between cities
distance_matrix = np.array([[0, 2, 9, 10],
                            [1, 0, 6, 4],
                            [15, 7, 0, 8],
                            [6, 3, 12, 0]])

# Create an instance of the AntColonyOptimization class
aco = AntColonyOptimization(num_ants=10, num_iterations=100, alpha=1, beta=2, rho=0.5, Q=100)

# Solve the TSP using the ACO algorithm
best_path, best_distance = aco.solve(distance_matrix)

# Print the best path and its distance
print("Best Path:", best_path)
print("Best Distance:", best_distance)
