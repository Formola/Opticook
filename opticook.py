import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import random as r
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter
from sklearn.preprocessing import MinMaxScaler

"""Constants"""

PRIMI_FILE_PATH      = "primi_official.json"
SECONDI_FILE_PATH    = "secondi_official.json"
FRIDGE_FILE_PATH     = "fridge.json"
C = 1000.0

"""NSGA implementation"""

# @title NSGA Implementation
class NSGA2Utils:

    def __init__(self, problem, num_of_individuals=100,
                 num_of_tour_particips=2, tournament_prob=0.9, crossover_param=2, mutation_param=5):

        self.problem = problem
        self.num_of_individuals = num_of_individuals
        self.num_of_tour_particips = num_of_tour_particips
        self.tournament_prob = tournament_prob
        self.crossover_param = crossover_param
        self.mutation_param = mutation_param

    def create_initial_population(self):
        population = Population()

        f1_values = []  # Lista per memorizzare i valori di f1
        f2_values = []  # Lista per memorizzare i valori di f2
        f3_values = []


        while len(population) < self.num_of_individuals:
            individual = self.problem.generate_individual()
            self.problem.calculate_objectives(individual)

            # Controllo se l'individuo è già presente nella popolazione, se non è presente, viene aggiunto nella popolazione, altrimenti viene rigenerato
            if individual not in population:
                population.append(individual)
                f1_values.append(individual.objectives[0])
                f2_values.append(individual.objectives[1])
                f3_values.append(individual.objectives[2])

        print("Grafici popolazione iniziale")
        plt.scatter(f1_values, f2_values)
        plt.title('Initial Population Objectives')
        plt.xlabel('Objective f1')
        plt.ylabel('Objective f2')
        plt.show()

        plt.scatter(f1_values, f3_values)
        plt.title('Initial Population Objectives')
        plt.xlabel('Objective f1')
        plt.ylabel('Objective f3')
        plt.show()

        plt.scatter(f2_values, f3_values)
        plt.title('Initial Population Objectives')
        plt.xlabel('Objective f2')
        plt.ylabel('Objective f3')
        plt.show()

        return population


    # Funzione di ordinamento veloce per trovare i fronti di pareto all'interno della popolazione
    def fast_nondominated_sort(self, population):
        population.fronts = [[]]
        unique_individuals = set()  # Set per tracciare individui unici nel fronte di Pareto

        for individual in population:
            individual.domination_count = 0
            individual.dominated_solutions = []


            # Verifica se l'individuo è già presente nel fronte di Pareto
            if individual not in unique_individuals and individual not in population.fronts:
                for other_individual in population:
                    if individual.dominates(other_individual):
                        individual.dominated_solutions.append(other_individual)   # le soluzione dominate da quell'individuo
                    elif other_individual.dominates(individual):
                        individual.domination_count += 1   # da quante soluzioni è dominato l'individuo
                if individual.domination_count == 0:    # se l'individuo non è dominato da nessuna soluzione, verrà inserito nel fronts[0], che sarà il fronte di pareto.
                    individual.rank = 0
                    population.fronts[0].append(individual)

                # Aggiungi l'individuo al set di individui unici
                unique_individuals.add(individual)

        i = 0
        while len(population.fronts[i]) > 0:
            temp = []
            for individual in population.fronts[i]:
                for other_individual in individual.dominated_solutions:
                    other_individual.domination_count -= 1
                    if other_individual.domination_count == 0:
                        other_individual.rank = i + 1
                        temp.append(other_individual)
            i = i + 1
            population.fronts.append(temp)

    # funzione che viene chiamata per ogni fronte generato
    def calculate_crowding_distance(self, front):
        if len(front) > 0:
            solutions_num = len(front)
            for individual in front:
                individual.crowding_distance = 0

            for m in range(len(front[0].objectives)):
                front.sort(key=lambda individual: individual.objectives[m])
                front[0].crowding_distance = 10 **9   # valori agli estremi del fronte alto, dice che è una pratica comune in nsga, non sono tanto d'accordo ma vabene per ora
                front[solutions_num - 1].crowding_distance = 10 ** 9
                m_values = [individual.objectives[m] for individual in front]
                scale = max(m_values) - min(m_values)
                if scale == 0: scale = 1
                for i in range(1, solutions_num - 1):
                    front[i].crowding_distance += (front[i + 1].objectives[m] - front[i - 1].objectives[m]) / scale    # differenza normalizzata tra i valori degli obiettivi di due individui adiacenti

    def crowding_operator(self, individual, other_individual):
        if (individual.rank < other_individual.rank) or \
                ((individual.rank == other_individual.rank) and (
                        individual.crowding_distance > other_individual.crowding_distance)):
            return 1
        else:
            return -1

    def create_children(self, population):
        children = []
        while len(children) < len(population):
            parent1 = self.__tournament(population)
            parent2 = parent1
            while parent1 == parent2:
                parent2 = self.__tournament(population)   # selezione tramite torneo binario dei due genitori
            child1, child2 = self.__crossover_brazilian(parent1, parent2)
            self.__mutate_brazilian(child1)
            self.__mutate_brazilian(child2)
            self.problem.calculate_objectives(child1)
            self.problem.calculate_objectives(child2)

            # Verifica se i figli sono già presenti nella popolazione
            if child1 not in children and child1 not in population:
                children.append(child1)

            if child2 not in children and child2 not in population:
                children.append(child2)

        return children

    def __crossover(self, individual1, individual2):
        child1 = self.problem.generate_individual()
        child2 = self.problem.generate_individual()
        num_of_features = len(child1.features)
        genes_indexes = range(num_of_features)
        # for i in genes_indexes:
        #     beta = self.__get_beta()
        #     x1 = (individual1.features[i] + individual2.features[i]) / 2
        #     x2 = abs((individual1.features[i] - individual2.features[i]) / 2)
        #     child1.features[i] = round(x1 + beta * x2)
        #     child2.features[i] = round(x1 - beta * x2)
        #     # child1.features[i] = x1 + beta * x2
        #     # child2.features[i] = x1 - beta * x2
        return print(self.__get_beta())


    def __crossover_brazilian(self, individual1, individual2):
        # TODO: x: primo, y: secondo  [ x1,y1 e x2,y2 => x1,y2, x2,y1 ] . Crossover proposto dai BRASILIANI. penso dovremmo fare ==>   x1,y2   x2,y1,
        child1 = self.problem.generate_individual()
        child2 = self.problem.generate_individual()
        # num_of_features = len(child1.features)
        # genes_indexes = range(num_of_features)

        # print("individuo1: ")
        # print(individual1.features)
        # print("individuo2: ")
        # print(individual2.features)


        # esempio brazilian: switchiamo le recipes nel meal
        # genotipo1:
          # pasta e patate --> individual1.features[0]
          # pollo --> individual1.features[1]
        #genotipo2:
          # pasta e zucchine  --> individual2.features[0]
          # polpette al sugo   --> individual2.features[1]
        # child1:
          # pasta e patate
          # polpette al sugo
        # child2:
          # pasta e zucchine
          # pollo

          # .....oppure.....

        # child1:
          # pasta e zucchine
          # pollo
        # child2:
          # pasta e patate
          # polpette al sugo



        # Caso 1: child1 prende la prima feature da individual1 e la seconda da individual2
        #         child2 prende la prima feature da individual2 e la seconda da individual1
        if random.random()*self.crossover_param <= (self.crossover_param/2):
            child1.features[0] = individual1.features[0]
            child1.features[1] = individual2.features[1]
            child2.features[0] = individual2.features[0]
            child2.features[1] = individual1.features[1]
        # Caso 2: child1 prende la prima feature da individual2 e la seconda da individual1
        #         child2 prende la prima feature da individual1 e la seconda da individual2
        else:
            child1.features[0] = individual2.features[0]
            child1.features[1] = individual1.features[1]
            child2.features[0] = individual1.features[0]
            child2.features[1] = individual2.features[1]

        # print("child1: ")
        # print(child1.features)
        # print("child2: ")
        # print(child2.features)



        return child1, child2

    def __get_beta(self):
        u = random.random()
        if u < 0.5:
            return (2 * u) ** (1 / (self.crossover_param + 1))
        else:
            return (2 * (1 - u)) ** (-1 / (self.crossover_param + 1))



    def __mutate(self, child):
        # DA CAMBIARE CON QUELLO DEI BRASILIANI)
        num_of_features = len(child.features)
        for gene in range(num_of_features):
            u, delta = self.__get_delta()
            if u < 0.5:
                # child.features[gene] += delta * (child.features[gene] - self.problem.variables_range[gene][0])
                child.features[gene] += round(delta * (child.features[gene] - self.problem.variables_range[gene][0]))
            else:
                child.features[gene] += round(delta * (self.problem.variables_range[gene][1] - child.features[gene]))
            if child.features[gene] < self.problem.variables_range[gene][0]:
                child.features[gene] = round(self.problem.variables_range[gene][0])
            elif child.features[gene] > self.problem.variables_range[gene][1]:
                child.features[gene] = round(self.problem.variables_range[gene][1])

    def __mutate_brazilian(self, child):
      min_value_0, max_value_0 = self.problem.variables_range[0]
      min_value_1, max_value_1 = self.problem.variables_range[1]

      # print("child before mutation: ")
      # print(child.features)

      #genotipo:
        # pasta e patate
        # pollo
      # genotipo after mutation:
        # lasagna
        # pollo

      u, delta = self.__get_delta()

      if u*self.mutation_param < (self.mutation_param/2): # sostutiamo la prima recipes ad esempio
        child.features[0] = random.randint(min_value_0, max_value_0)
      else:
          child.features[1] = random.randint(min_value_1, max_value_1)  # sostituiamo la seconda recipes

      # print("child after mutation: ")
      # print(child.features)




    def __get_delta(self):
        u = random.random()
        if u < 0.5:
            return u, (2 * u) ** (1 / (self.mutation_param + 1)) - 1
        return u, 1 - (2 * (1 - u)) ** (1 / (self.mutation_param + 1))

    def __tournament(self, population):
        participants = random.sample(population.population, self.num_of_tour_particips)
        best = None
        for participant in participants:
            if best is None or (
                    self.crowding_operator(participant, best) == 1 and self.__choose_with_prob(self.tournament_prob)):
                best = participant

        return best

    def __choose_with_prob(self, prob):
        if random.random() <= prob:
            return True
        return False

class Problem:

    def __init__(self, num_of_variables, objectives, variables_range, expand=True, same_range=False):
        self.num_of_objectives = len(objectives)
        self.num_of_variables = num_of_variables
        self.objectives = objectives
        self.expand = expand
        self.variables_range = []
        if same_range:
            for _ in range(num_of_variables):
                self.variables_range.append(variables_range[0])
        else:
            self.variables_range = variables_range

    def generate_individual(self):
        individual = Individual()
        individual.features = [round(random.uniform(*x)) for x in self.variables_range]
        return individual

    def calculate_objectives(self, individual):
        if self.expand:
            individual.objectives = [f(*individual.features) for f in self.objectives]
        else:
            individual.objectives = [f(individual.features) for f in self.objectives]

class Population:

    def __init__(self):
        self.population = []
        self.fronts = []

    def __len__(self):
        return len(self.population)

    def __iter__(self):
        return self.population.__iter__()

    def extend(self, new_individuals):
        self.population.extend(new_individuals)

    def append(self, new_individual):
        self.population.append(new_individual)

class Individual(object):

    def __hash__(self):
        return hash(tuple(sorted(self.features)))

    def __init__(self):
        self.rank = None
        self.crowding_distance = None
        self.domination_count = None
        self.dominated_solutions = None
        self.features = None
        self.objectives = None

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.features == other.features
        return False

    # applicchiamo la definizione di dominanza
    def dominates(self, other_individual):
        and_condition = True
        or_condition = False
        for first, second in zip(self.objectives, other_individual.objectives):
            # La and condition deve essere verificata su TUTTI gli OBJECTIVES
            # Sta dicendo che l'individuo non peggiora nessun altro obiettivo rispetto al secondo individuo
            and_condition = and_condition and first <= second

            # La or-condition deve essere verificata su ALMENO UN Obiettivo
            # E ne migliora ALMENO uno
            or_condition = or_condition or first < second
        return (and_condition and or_condition)

class Evolution:

    def __init__(self, problem, num_of_generations=100, num_of_individuals=100, num_of_tour_particips=2,
                 tournament_prob=0.9, crossover_param=2, mutation_param=5):
        self.utils = NSGA2Utils(problem, num_of_individuals, num_of_tour_particips, tournament_prob, crossover_param,
                                mutation_param)
        self.population = None
        self.num_of_generations = num_of_generations
        self.on_generation_finished = []
        self.num_of_individuals = num_of_individuals

    def evolve(self):
        self.population = self.utils.create_initial_population()
        self.utils.fast_nondominated_sort(self.population)
        self.plot_pareto_fronts()
        for front in self.population.fronts:
            self.utils.calculate_crowding_distance(front)
        children = self.utils.create_children(self.population)
        returned_population = None
        for i in tqdm(range(self.num_of_generations)):
            self.population.extend(children)
            self.utils.fast_nondominated_sort(self.population)
            new_population = Population()
            front_num = 0

            # blocco che condensa gli individui nei fronti successivi nel fronte precedente se non raggiunge il numero di individui fissato
            while front_num < len(self.population.fronts) and len(new_population) + len(self.population.fronts[front_num]) <= self.num_of_individuals:
                self.utils.calculate_crowding_distance(self.population.fronts[front_num])
                new_population.extend(self.population.fronts[front_num])
                front_num += 1

            if front_num < len(self.population.fronts):
                self.utils.calculate_crowding_distance(self.population.fronts[front_num])
                self.population.fronts[front_num].sort(key=lambda individual: individual.crowding_distance, reverse=True)
                new_population.extend(self.population.fronts[front_num][0:self.num_of_individuals - len(new_population)])

            returned_population = self.population
            self.population = new_population
            self.utils.fast_nondominated_sort(self.population)
            for front in self.population.fronts:
                self.utils.calculate_crowding_distance(front)
            children = self.utils.create_children(self.population)

        self.plot_pareto_fronts()

        return returned_population.fronts[0]

    def plot_pareto_fronts(self):
        plt.figure(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.population.fronts)))

        for i, front in enumerate(self.population.fronts):
            x = [individual.objectives[0] for individual in front]
            y = [individual.objectives[1] for individual in front]
            label = f'Front {i + 1}'
            plt.scatter(x, y, label=label, color=colors[i])

        plt.title('Pareto Fronts')
        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')
        plt.legend()
        plt.show()

# @title Opticook
with open(PRIMI_FILE_PATH, "r", encoding="utf-8") as file:
    primi_json = json.load(file)

with open(SECONDI_FILE_PATH, "r", encoding="utf-8") as file:
    secondi_json =json.load(file)

with open(FRIDGE_FILE_PATH, "r", encoding="utf-8") as file:
    fridge_json =json.load(file)

def f1(x, y):
    x = int(x)
    y = int(y)
    c = np.array([primi_json[x]["nutrients"]["calories"], secondi_json[y]["nutrients"]["calories"]])
    return np.abs(1 -(np.sum(c) / C))

def f2(x, y):
    x = int(x)
    y = int(y)
    q1 = np.array([ingredient["quantity"] for ingredient in primi_json[x]["ingredients"]])
    q2 = np.array([ingredient["quantity"] for ingredient in secondi_json[y]["ingredients"]])
    return  ( 1 /  (np.sum(q1) + np.sum(q2)) )


def f3(x, y):
    x = int(x)
    y = int(y)
    D_q1 = []
    D_q2 = []
    for ingredient_data in primi_json[x]["ingredients"]:
      ingredient_name = ingredient_data["ingredient"]
      if ingredient_name in fridge_json:
        D_q1.append(fridge_json[ingredient_name]["remaining_time"])
      else:
        D_q1.append(100)

    for ingredient_data in secondi_json[y]["ingredients"]:
      ingredient_name = ingredient_data["ingredient"]
      if ingredient_name in fridge_json:
        D_q2.append(fridge_json[ingredient_name]["remaining_time"])
      else:
        D_q2.append(100)

    D_q1 = np.array(D_q1)
    D_q2 = np.array(D_q2)

    return (np.sum(D_q1**2) + np.sum(D_q2**2))


problem = Problem(num_of_variables=2, objectives=[f1,f2,f3], variables_range=[(0, len(primi_json) - 1), (0, len(secondi_json) - 1)])
evo = Evolution(problem)
evol = evo.evolve()
func = [i.objectives for i in evol]
function1 = [i[0] for i in func]
function2 = [i[1] for i in func]
function3 = [i[2] for i in func]

scaler = MinMaxScaler()
normalized_func = scaler.fit_transform(func)
normalized_function1 = normalized_func[:, 0]
normalized_function2 = normalized_func[:, 1]
normalized_function3 = normalized_func[:, 2]

plt.figure(figsize=(10, 6))
plt.scatter(function1, function2, label='Pareto Front')
plt.title('Pareto Front: f1 vs f2')
plt.xlabel('Function 1')
plt.ylabel('Function 2')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(normalized_function1, normalized_function3, label='Pareto Front')
plt.title('Pareto Front: f1 vs f3')
plt.xlabel('Function 1')
plt.ylabel('Function 3')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(normalized_function2, normalized_function3, label='Pareto Front')
plt.title('Pareto Front: f2 vs f3')
plt.xlabel('Function 2')
plt.ylabel('Function 3')
plt.legend()
plt.show()

# Crea il grafico tridimensionale
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Grafica i punti
scatter = ax.scatter(normalized_function1, normalized_function2, normalized_function3)

# Aggiungi etichette
ax.set_xlabel('Function 1', fontsize=15)
ax.set_ylabel('Function 2', fontsize=15)
ax.set_zlabel('Function 3', fontsize=15)

# Mostra il grafico
plt.show()


features_indices = [tuple(i.features) for i in evol]
primi_results = [primi_json[idx[0]] for idx in features_indices]
secondi_results = [secondi_json[idx[1]] for idx in features_indices]
meal_ottimi = []

for primo, secondo in zip(primi_results, secondi_results):
    meal_ottimo = {"primo": primo, "secondo": secondo}
    meal_ottimi.append(meal_ottimo)

print(f"Ci sono {len(meal_ottimi)} meal ottimi!\n")

def calculate_total_quantities(primo, secondo):
    total_quantities = {}

    def add_quantity(ingredient_name, ingredient_quantity):
        if ingredient_name in total_quantities:
            total_quantities[ingredient_name] += ingredient_quantity
        else:
            total_quantities[ingredient_name] = ingredient_quantity

    # Calcola le quantità totali per il primo piatto
    for ingredient_data in primo["ingredients"]:
        ingredient_name = ingredient_data["ingredient"]
        ingredient_quantity = ingredient_data["quantity"]
        add_quantity(ingredient_name, ingredient_quantity)

    # Calcola le quantità totali per il secondo piatto
    for ingredient_data in secondo["ingredients"]:
        ingredient_name = ingredient_data["ingredient"]
        ingredient_quantity = ingredient_data["quantity"]
        add_quantity(ingredient_name, ingredient_quantity)

    return total_quantities

# Calcola le quantità totali per ogni "meal ottimo"
total_quantities_list = [calculate_total_quantities(meal["primo"], meal["secondo"]) for meal in meal_ottimi]

# Stampa le quantità totali per ciascun "meal ottimo"
for idx, total_quantities in enumerate(total_quantities_list, start=1):
    print(f"\nQuantità totali per il Meal {idx}:")
    for ingredient, quantity in total_quantities.items():
        print(f"{ingredient}: {quantity}g")


def check_recipe_availability(primo, secondo):
    total_quantities = calculate_total_quantities(primo, secondo)

    for ingredient, required_quantity in total_quantities.items():
        if ingredient in fridge_json:
            fridge_quantity = fridge_json[ingredient]["quantity"]

            # Controlla la quantità di ciascun ingrediente
            if fridge_quantity < required_quantity*0.5 :
                return False
        else:
            # L'ingrediente non è presente nel frigorifero
            return False

    # Tutti gli ingredienti sono disponibili in quantità sufficiente nel frigorifero
    return True

# Verifica la disponibilità dei meal nel frigorifero
available_meals = [meal for meal in meal_ottimi if check_recipe_availability(meal["primo"], meal["secondo"])]


# all_ingredients = []

# for meal in meal_ottimi:
#     primo_ingredients = meal["primo"].get("ingredients", [])
#     secondo_ingredients = meal["secondo"].get("ingredients", [])

#     # Aggiungi gli ingredienti solo se sono presenti nella chiave "ingredients"
#     all_ingredients.extend([(ingredient["ingredient"], ingredient["quantity"]) for ingredient in primo_ingredients])
#     all_ingredients.extend([(ingredient["ingredient"], ingredient["quantity"]) for ingredient in secondo_ingredients])

# # Calcola la frequenza di ciascun ingrediente
# ingredient_frequencies = Counter(all_ingredients)

# # Trova gli ingredienti più comuni
# most_common_ingredients = ingredient_frequencies.most_common()

# # Stampa gli ingredienti più comuni
# print("\nIngredienti più comuni nei meal ottimi:")
# for (ingredient, quantity), frequency in most_common_ingredients:
#     print(f"{ingredient}: {quantity}g - {frequency} volte")

for meal in available_meals:
  ingredient_quantities = {}
  primo = meal['primo']
  secondo = meal['secondo']
  for ingredient_obj in primo["ingredients"]:
    if ingredient_obj['ingredient'] in ingredient_quantities:
      ingredient_quantities[ingredient_obj["ingredient"]] += ingredient_obj['quantity']
    else:
      ingredient_quantities[ingredient_obj['ingredient']] = ingredient_obj['quantity']
  for ingredient_obj in secondo["ingredients"]:
    if ingredient_obj['ingredient'] in ingredient_quantities:
      ingredient_quantities[ingredient_obj["ingredient"]] += ingredient_obj['quantity']
    else:
      ingredient_quantities[ingredient_obj['ingredient']] = ingredient_obj['quantity']

  for ingredient in ingredient_quantities:
    ingredient_quantities[ingredient] = {
        "quantity": ingredient_quantities[ingredient],
        "price": round(ingredient_quantities[ingredient] * (fridge_json[ingredient]["price"] / 1000),2)
      }

  meal['total_price'] = round(sum([ingredient_quantities[ingredient]["price"] for ingredient in ingredient_quantities]),2)

print(f"\nCon gli ingredienti nel tuo frigo puoi cucinare i seguenti meal ottimi ({len(available_meals)} meal ottimi)\n")


for idx, meal in enumerate(available_meals, start=1):
    print(f"Meal {idx}:\nPrimo: {meal['primo']['title']}\nSecondo: {meal['secondo']['title']}")
    print(f"Prezzo totale: {meal['total_price']}\n")


sorted_meals = sorted(available_meals, key=lambda x: x['total_price'])
# print([round(meal['total_price'],2) for meal in sorted_meals])
print("\nBEST MEAL: ")
print(f"Primo: {sorted_meals[0]['primo']['title']}\nSecondo: {sorted_meals[0]['secondo']['title']}\nPrezzo totale: {sorted_meals[0]['total_price']}")

