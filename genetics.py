import random
from PIL import Image, ImageDraw
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import copy


OFFSET = 10


def generate_point(width, height):
    x = random.randrange(0 - OFFSET, width + OFFSET, 1)
    y = random.randrange(0 - OFFSET, height + OFFSET, 1)
    return (x, y)

class Triangle:
    def __init__(self, img_width, img_height):
        self.points = []
        for i in range(3):
            self.points.append(generate_point(img_width,img_height))
        
        self.color_r = random.randint(0, 255)
        self.color_g = random.randint(0, 255)
        self.color_b = random.randint(0, 255)
        self.color_a = random.randint(0, 255)
        self.color = (self.color_r, self.color_g, self.color_b, self.color_a)

        self._img_width = img_width
        self._img_height = img_height
    
    def change_color_slightly(self):
        self.color_r = min(255, max(0, self.color_r + random.randint(-15, 15)))
        self.color_g = min(255, max(0, self.color_g + random.randint(-15, 15)))
        self.color_b = min(255, max(0, self.color_b + random.randint(-15, 15)))
        self.color_a = min(255, max(0, self.color_a + random.randint(-15, 15)))

    def change_color(self):
        colors = list(self.color)
        colors[random.randint(0, 3)] = random.randint(0, 255)
        self.color = tuple(colors)

    def change_position(self):
        self.points[random.randint(0, 2)] = generate_point(self._img_width, self._img_height)

    def change_position_slightly(self):
        self.points = [(x + random.uniform(-4, 4), y + random.uniform(-4, 4)) for (x, y) in self.points]

    def change_size_slightly(self):
        factor = random.uniform(0.9, 1.1)
        self.points = [(x * factor, y * factor) for (x, y) in self.points]
        
        


    class Chromosome:  
    def __init__(self,img_height, img_width,_target_image, _num_triangles, _triangles):
        self.img_height = img_height
        self.img_width = img_width  
        self.background_color = (0,0,0,255)
        self.triangles = _triangles
        self.target_image = _target_image
        self.num_triangles = _num_triangles
        self.fitness_score = 0

    def mutate(self):

        triangle = random.choice(self.triangles)
        mutation_type = random.choice(['position', 'color', 'both'])

        if mutation_type == 'position':
            triangle.change_position()
        
        elif mutation_type == 'color':
            triangle.change_color()
        
        elif mutation_type == 'both':
            triangle.change_color()
            triangle.change_position()
    
    def draw(self) -> Image:
        size = self.target_image.size
        img = Image.new('RGB', size, self.background_color)
        draw = Image.new('RGBA', size)
        pdraw = ImageDraw.Draw(draw)
        for triangle in self.triangles:
            colour = triangle.color
            points = triangle.points
            pdraw.polygon(points, fill=colour, outline=colour)
            img.paste(draw, mask=draw)
        return img
        
    def fitness(self) -> float:
        created_image = np.array(self.draw())
        mse = np.sum((created_image/255 - np.array(self.target_image)/255) ** 2)
        fitness_value = -mse
        self.fitness_score = fitness_value
        return fitness_value

class GeneticAlgorithm():
    def __init__(self,max_width,max_height,target_image, population_size, _triangles_number):
        self.population_size = population_size
        self.max_width = max_width
        self.max_height = max_height
        self.population = [Chromosome(max_height,max_width,target_image, _triangles_number, [Triangle(max_width,max_height) for i in range(_triangles_number)]) for i in range(population_size)]
        self.target_image = target_image
        self.generation = 0
        self.triangles_number = _triangles_number
        self.elite_count = (int)(population_size * 0.3)
        self.pm = 0.3
    
    def calc_fitnesses(self):
        fitnesses = []
        for chromosome in self.population:  
            fitnesses.append(chromosome.fitness())
        #print(fitnesses)
        return fitnesses
    
    def sort_population(self, fitnesses):
        return [x for _, x in sorted(zip(fitnesses, self.population), key=lambda pair: pair[0])]
    
    def crossover(self):
        for i in range(self.population_size // 2):
            if random.random() < 0.8:
                parent1, parent2 = random.choices(self.population, k = 2)
                child1_triangles, child2_triangles = self.uniform_crossover(parent1, parent2)
                parent1.triangles = child1_triangles
                parent2.triangles = child2_triangles
                parent1.fitness()
                parent2.fitness()
        
    def uniform_crossover(self, parent1, parent2):
        child1_triangles = []
        child2_triangles = []

        for i in range(self.triangles_number):
            if random.random() > 0.5:
                child1_triangles.append(parent1.triangles[i])
                child2_triangles.append(parent2.triangles[i])
            else:
                child1_triangles.append(parent2.triangles[i])
                child2_triangles.append(parent1.triangles[i])

        return child1_triangles, child2_triangles
 
    def single_point_crossover(self, parent1, parent2):
        crossover_point = random.randint(1, len(parent1.triangles) - 1)

        child1_triangles = parent1.triangles[:crossover_point] + parent2.triangles[crossover_point:]
        child2_triangles = parent2.triangles[:crossover_point] + parent1.triangles[crossover_point:]

        return child1_triangles, child2_triangles
    
    def two_point_crossover(self, parent1, parent2):
        point1 = random.randint(1, len(parent1.triangles) - 2)
        point2 = random.randint(point1 + 1, len(parent1.triangles) - 1)

        child1_triangles = parent1.triangles[:point1] + parent2.triangles[point1:point2] + parent1.triangles[point2:]
        child2_triangles = parent2.triangles[:point1] + parent1.triangles[point1:point2] + parent2.triangles[point2:]

        return child1_triangles, child2_triangles
    
    def adaptive_tournament_selection(self):
        initial_tournament_size = 8
        max_tournament_size = 12
        current_tournament_size = min(initial_tournament_size + (self.generation // 50), max_tournament_size)
        parents = []

        for _ in range(len(self.population) - self.elite_count):
            tournament = random.sample(self.population, current_tournament_size)
            
            selected_parent = max(tournament, key=lambda chromosome: chromosome.fitness())
            parents.append(selected_parent)
        return parents
    
    def roulette_wheel_selection(self):
        fitnesses = [chromosome.fitness_score for chromosome in self.population]
        total_fitness = sum(fitnesses)
        selection_probs = [fitness / total_fitness for fitness in fitnesses]

        selected_population = random.choices(self.population, weights=selection_probs, k=self.population_size - self.elite_count)
        return selected_population
    
    def mutation(self):  
        for chromosome in self.population:
            if(random.random() < self.pm):
                chromosome.mutate()
                chromosome.fitness()
    
    def update_pm(self):
        self.pm = max(0.2, self.pm - (self.generation // 400) * 0.02)

    def run(self,n_generations):
        fitnesses = self.calc_fitnesses()
        for iteration in range(n_generations):
            self.update_pm()

            elite = copy.deepcopy(sorted(self.population, key=lambda chromo: chromo.fitness_score, reverse=True)[:self.elite_count])

            self.crossover()
            self.mutation()

            new_population = (sorted(self.population, key=lambda chromo: chromo.fitness_score, reverse=True)[:self.population_size - self.elite_count])
            #new_population = self.roulette_wheel_selection()
            self.population = (new_population + elite)

            if iteration % 20 == 0:
                #this part shows the log for fitness
                fitnesses = [chromosome.fitness_score for chromosome in self.population]
                fit_arr = np.array(fitnesses)
                print(f"Fitness in Generation {iteration}: mean: {np.mean(fit_arr)}, max: {np.max(fit_arr)} min: {np.min(fit_arr)}")
                
                if iteration % 50 == 0:
                    best_chromosome = max(self.population, key=lambda chromo: chromo.fitness_score)
                    best_image = best_chromosome.draw()
                    self.display_image(best_image) 
                    
                    #self.get_best_of_population()  

    def display_image(self, image):
        plt.imshow(image)
        plt.title("Best Image in Generation")
        plt.axis('off')
        plt.show()
            
    def get_best_of_population(self):
        fitnesses = self.calc_fitnesses()
        sorted_population = [x for _, x in sorted(zip(fitnesses, self.population), key=lambda pair: pair[0])]              
        best_population = sorted_population[-1]
        image = best_population.draw()       
        cv2.imshow("Reconstructed Image", np.array(image))  
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def resize(image,max_size):
    new_width = int((max_size/max(image.size[0],image.size[1]))* image.size[0])
    new_height = int((max_size/max(image.size[0],image.size[1]))* image.size[1])
    image = image.resize((new_width,new_height), resample=Image.Resampling.LANCZOS)  
    return image

target_image_path = r"C:\Users\Mahmodiyan-PC\Desktop\eagle.jpg"
image = Image.open(target_image_path)
# Use resize to resize your images
image = resize(image,100)

width,height = image.size
population_size = 50
triangles_number = 50
alg = GeneticAlgorithm(width,height,image, population_size,triangles_number)
alg.run(50000)