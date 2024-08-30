import ast
from tqdm import tqdm
import subprocess
from circuit_generation import *


from copy import deepcopy
import os
import shutil
import importlib.util
from qiskit.circuit.exceptions import CircuitError
from qiskit import QuantumCircuit, Aer, transpile
from qiskit.quantum_info import Operator, state_fidelity,process_fidelity
import Levenshtein
import time
from tqdm import tqdm
rotation_gates = ['rx', 'ry', 'rz', 'u1', 'u2', 'u3', 'crx', 'cry', 'crz', 'cp', 'cu1', 'cu3']
multi_qubit_gates = ['cx', 'cz', 'swap', 'ch', 'csx', 'cy', 'ccx', 'cswap', 'cu', 'cp']
three_qubit_gates = ['ccx', 'cswap']  # Toffoli (CCX) gate and Fredkin (CSWAP) gate


def analyze_ast(node, output=False):
    gate_calls = []
    qc_calls = []
    parent_info = []
    index_depths = []

    def visit_node(node, depth=0):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'qc':
            qc_calls.append(node)
            index_depth = 0
            args = [ast.unparse(arg) for arg in node.args]
            if output:
                print(f"{'  ' * depth}Found qc call: {ast.dump(node)} at depth {depth}")
                print(f"{'  ' * depth}Arguments: {args}")
            for arg in node.args:
                arg_str = ast.unparse(arg)
                if 'pi' in arg_str:
                    continue
                # Check for 'i' and find the maximum number following 'i'
                for part in arg_str.split():
                    if 'i' in part:
                        i_pos = part.find('i')
                        if i_pos != -1 and i_pos < len(part) - 1:
                            num_str = ''.join(filter(str.isdigit, part[i_pos+1:]))
                            if num_str:
                                index_depth = max(index_depth, int(num_str) + 1)
            index_depths.append(index_depth)
            if output:
                print(f"{'  ' * depth}Index Depth: {index_depth}\n")
        for child in ast.iter_child_nodes(node):
            visit_node(child, depth + 1)

    visit_node(node)
    return gate_calls, qc_calls, parent_info, index_depths






class genetic_Decompiler:
    def __init__(self, algorithm_name, qubit_limit=20, generations=100, pop_size=50, max_length=10, 
                 perform_crossover=True,crossover_rate=0.3, new_gen_rate=0.2,mutation_rate=0.1,
                 compare_method='l_by_l',max_loop_depth=2, mutation_rate_2=0.5, perform_annealing=False,
                perform_mutation=True, selection_method='tournament',operations = ['h', 'x', 'cx']):
        self.algorithm_name = algorithm_name
        self.qubit_limit = qubit_limit
        self.generations = generations
        self.pop_size = pop_size
        self.max_length = max_length
        self.crossover_rate=crossover_rate
        self.mutation_rate=mutation_rate
        self.mutation_rate_2 = mutation_rate_2
        self.new_gen_rate=new_gen_rate
        self.max_loop_depth=max_loop_depth
        self.perform_crossover = perform_crossover
        self.compare_method=compare_method
        self.perform_annealing = perform_annealing
        self.perform_mutation = perform_mutation
        self.selection_method = selection_method
        self.operations=operations
        # Initialize the path for saving files related to the algorithm
        self.path = os.path.join('genetic_deQ', self.algorithm_name)
        self.qasm_path=os.path.join('genetic_deQ_qasm', self.algorithm_name)
        os.makedirs(self.path, exist_ok=True)  # Create the directory if it does not exist
        os.makedirs(self.qasm_path, exist_ok=True) 
 

    def generate_initial_population(self,size):
        population = []
        for _ in range(size):
            # num_qubits = random.randint(2, self.qubit_limit)
            num_nodes = random.randint(1, self.max_length)
            ast_circuit = generate_random_circuit_ast( num_nodes,self.operations,max_loop_depth=self.max_loop_depth)
            population.append(ast_circuit)
        return population

    
    def mutate(self, ast_circuit, mutation_rate_2=0.5, output=False):
        mutation_rate_2 = self.mutation_rate_2
        # Create a deep copy of the AST to avoid modifying the original AST
        ast_circuit_copy = deepcopy(ast_circuit)

        # Analyze the AST
        gate_calls, qc_calls, parent_info, index_depths = analyze_ast(ast_circuit_copy, output=False)

        if not qc_calls:
            return ast_circuit_copy  # No gate calls to mutate

        # Randomly choose mutation type
        mutation_type = random.choices(['insert', 'modify'], weights=[0.2, 0.8])[0]

        if mutation_type == 'insert':
            # Randomly select a parent node to insert into
            if not parent_info:
                return ast_circuit_copy  # No parent nodes to insert into

            parent_node, parent_index = random.choice(parent_info)
            new_gate = generate_gate_call(random.choice(self.operations))
            parent_node.body.insert(parent_index, new_gate)
            if output:
                print(f"Inserted new gate: {ast.unparse(new_gate)} at index {parent_index}")

        elif mutation_type == 'modify':
            # Randomly select a qc call to mutate with a probability
            for qc_call, index_depth in zip(qc_calls, index_depths):
                if random.random() < mutation_rate_2:
                    if output:
                        original_code = ast.unparse(qc_call)

                    # Extract arguments and classify them
                    for i, arg in enumerate(qc_call.args):
                        arg_str = ast.unparse(arg)
                        if 'pi' in arg_str:
                            # Mutate phase argument
                            qc_call.args[i] = random_phase_expr(index_depth)
                        else:
                            # Mutate index argument
                            new_expr = random_expr(index_depth, 3, 1)
                            qc_call.args[i] = random_qubit_expr(new_expr)

                    if output:
                        new_code = ast.unparse(qc_call)
                        print(f"Modified code from: {original_code} to: {new_code}")

        ast.fix_missing_locations(ast_circuit_copy)
        return ast_circuit_copy

    def crossover(self, parent1, parent2):
        # Select crossover points
        index1 = random.randint(1, len(parent1.body[0].body) - 2)
        index2 = random.randint(1, len(parent2.body[0].body) - 2)
        
        # Swap subcircuits
        new_body1 = parent1.body[0].body[:index1] + parent2.body[0].body[index2:]
        new_body2 = parent2.body[0].body[:index2] + parent1.body[0].body[index1:]
        
        # Construct new ASTs
        child1 = ast.Module(body=[ast.FunctionDef(
            name=parent1.body[0].name, 
            args=parent1.body[0].args, 
            body=new_body1, 
            decorator_list=[]
        )], type_ignores=[])
        
        child2 = ast.Module(body=[ast.FunctionDef(
            name=parent2.body[0].name, 
            args=parent2.body[0].args, 
            body=new_body2, 
            decorator_list=[]
        )], type_ignores=[])
        
        ast.fix_missing_locations(child1)
        ast.fix_missing_locations(child2)
        
        return child1, child2

    def select_parents(self, population, fitness_scores, selection_method='tournament', k=3):
        if selection_method == 'roulette':
            return self.roulette_wheel_selection(population, fitness_scores)
        elif selection_method == 'tournament':
            return self.tournament_selection(population, fitness_scores, k)
        elif selection_method == 'rank':
            return self.rank_selection(population, fitness_scores)
        elif selection_method == 'random':
            return self.random_selection(population)
        elif selection_method == 'weighted_roulette':
            return self.weighted_roulette_wheel_selection(population, fitness_scores)
        else:
            raise ValueError(f"Unknown selection method: {selection_method}")
    
    def roulette_wheel_selection(self, population, fitness_scores):
        total_fitness = sum(fitness_scores)
        probabilities = [score / total_fitness for score in fitness_scores]
        selected_indices = random.choices(range(len(population)), weights=probabilities, k=2)
        return population[selected_indices[0]], population[selected_indices[1]]

    def tournament_selection(self, population, fitness_scores, k=3):
        selected_indices = random.sample(range(len(population)), k)
        selected_individuals = [(fitness_scores[i], population[i]) for i in selected_indices]
        parent1 = max(selected_individuals, key=lambda x: x[0])[1]
        parent2 = max(selected_individuals, key=lambda x: x[0])[1]
        return parent1, parent2

    def rank_selection(self, population, fitness_scores):
        sorted_population = sorted(zip(fitness_scores, population), key=lambda x: x[0])
        rank_probabilities = [(i + 1) / len(sorted_population) for i in range(len(sorted_population))]
        selected_indices = random.choices(range(len(population)), weights=rank_probabilities, k=2)
        return sorted_population[selected_indices[0]][1], sorted_population[selected_indices[1]][1]

    def random_selection(self, population):
        parent1, parent2 = random.sample(population, 2)
        return parent1, parent2

    def weighted_roulette_wheel_selection(self, population, fitness_scores, weight=2.0):
        total_fitness = sum(fitness_scores)
        weighted_fitness = [score ** weight for score in fitness_scores]
        total_weighted_fitness = sum(weighted_fitness)
        probabilities = [wf / total_weighted_fitness for wf in weighted_fitness]
        selected_indices = random.choices(range(len(population)), weights=probabilities, k=2)
        return population[selected_indices[0]], population[selected_indices[1]]

    def save(self, population):
    # Clear all files in the target directory before saving new files
        for filename in os.listdir(self.path):
            file_path = os.path.join(self.path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove file
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove directory
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

        # Iterate over the population and save each individual's Python code to a file
        for index, individual in enumerate(population):
            # Convert AST to Python code
            python_code = ast.unparse(individual)

            # Create the filename, including the algorithm name and index
            filename = os.path.join(self.path, f"{self.algorithm_name}_{index}.py")
            
            # Write Python code to the file
            with open(filename, 'w') as file:
                file.write(python_code)

    def get_quantum_gates_from_qasm(self):
        target_qasm_dir = "Circuits"
        all_gates = set()

        for i in range(2, self.qubit_limit + 1):
            target_qasm_file = os.path.join(target_qasm_dir, f"{self.algorithm_name}_{i}.qasm")
            
            if os.path.exists(target_qasm_file):
                with open(target_qasm_file, 'r') as file:
                    qasm_str = file.read()
                
                quantum_circuit = QuantumCircuit.from_qasm_str(qasm_str)
                
                for instruction in quantum_circuit.data:
                    gate_name = instruction[0].name
                    all_gates.add(gate_name)
        
        return list(all_gates)

    def save_qasm(self):
        for filename in os.listdir(self.path):
            if filename.endswith('.py'):
                full_py_path = os.path.join(self.path, filename)
                
                # Read and modify the script as discussed above
                with open(full_py_path, 'r') as file:
                    module_code = "from qiskit import QuantumCircuit\nimport numpy as np\nimport random\nfrom math import pi\n" + file.read()

                local_namespace = {}
                exec(module_code, local_namespace)
                
                # Set up the directory for QASM files
                file_base_name = filename[:-3]  # Remove '.py' extension
                qasm_dir_path = os.path.join(self.qasm_path, file_base_name)
                os.makedirs(qasm_dir_path, exist_ok=True)
                
                # Generate QASM files for each qubit count
                for i in range(2, self.qubit_limit + 1):
                    try:
                            # Print the generated Python code for debugging
                        # generated_code = module_code + f"\n\ngenerate_random_circuit_ast({i})"
                        # print(f"Running generated code for {file_base_name} with {i} qubits:\n{generated_code}")


                        qc = local_namespace['generate_random_circuit_ast'](i)

                        modified_circuit = QuantumCircuit(qc.num_qubits)
                        for gate, qargs, cargs in qc.data:
                            if gate.name == 'cx':
                                control_qubit, target_qubit = qargs
                                if control_qubit.index == target_qubit.index:
                                    # Adjust target qubit index to be different from control qubit index
                                    target_qubit = qc.qubits[(target_qubit.index + 1) % qc.num_qubits]
                                    modified_circuit.cx(control_qubit, target_qubit)
                                else:
                                    modified_circuit.cx(control_qubit, target_qubit)
                            else:
                                modified_circuit.append(gate, qargs, cargs)
                        
                        qasm_output = modified_circuit.qasm()
                    except (CircuitError, ZeroDivisionError) as e:
                    # Handle both CircuitError and ZeroDivisionError
                    # print(f"Error generating QASM for {filename} with {i} qubits: {e}")
                        qasm_output = ""  # Save an empty QASM file if there's an error
                    
                    qasm_filename = os.path.join(qasm_dir_path, f"{file_base_name}_{i}.qasm")
                    with open(qasm_filename, 'w') as f:
                        f.write(qasm_output)

    
    def qasm_to_unitary(self, qasm_file_path):
        # Read QASM file and create a quantum circuit
        with open(qasm_file_path, 'r') as file:
            qasm_str = file.read()
        
        quantum_circuit = QuantumCircuit.from_qasm_str(qasm_str)
        
        # Use Aer simulator to get the unitary matrix
        backend = Aer.get_backend('unitary_simulator')
        transpiled_circuit = transpile(quantum_circuit, backend)
        
        # Get the unitary matrix
        job = backend.run(transpiled_circuit)
        unitary_matrix = job.result().get_unitary(transpiled_circuit)
        
        return unitary_matrix

    def qasm_to_gate_sequence(self, qasm_file_path):
        # Read QASM file and create a quantum circuit
        with open(qasm_file_path, 'r') as file:
            qasm_str = file.read()
        
        quantum_circuit = QuantumCircuit.from_qasm_str(qasm_str)
        
        # Extract gate sequence
        gate_sequence = []
        for instruction in quantum_circuit.data:
            gate_name = instruction[0].name
            qubits = [qubit.index for qubit in instruction[1]]
            if gate_name in rotation_gates:
                params = [param for param in instruction[0].params]
                gate_sequence.append((gate_name, tuple(qubits), params))
            else:
                gate_sequence.append((gate_name, tuple(qubits)))
        
        return gate_sequence
      
    def gate_sequence_similarity(self, seq1, seq2):
        seq1_str = ' '.join([f"{gate[0]}{gate[1]}{[f'{param:.6f}' for param in gate[2]]}" if len(gate) == 3 else f"{gate[0]}{gate[1]}" for gate in seq1])
        seq2_str = ' '.join([f"{gate[0]}{gate[1]}{[f'{param:.6f}' for param in gate[2]]}" if len(gate) == 3 else f"{gate[0]}{gate[1]}" for gate in seq2])
        
        ### Debugging line
        # print(f"Sequence 1: {seq1_str}")
        # print(f"Sequence 2: {seq2_str}")
        
        max_len = max(len(seq1_str), len(seq2_str))
        if max_len == 0:
            return 1.0
        
        return 1 - (Levenshtein.distance(seq1_str, seq2_str) / max_len)**(1/2)

    def gate_frequency_similarity(self, qasm_file_path1, qasm_file_path2):
        def get_gate_frequencies(qasm_file_path):
            with open(qasm_file_path, 'r') as file:
                qasm_str = file.read()
            
            quantum_circuit = QuantumCircuit.from_qasm_str(qasm_str)
            gate_count = {}
            for instruction in quantum_circuit.data:
                gate_name = instruction[0].name
                if gate_name in gate_count:
                    gate_count[gate_name] += 1
                else:
                    gate_count[gate_name] = 1
            return gate_count

        freq1 = get_gate_frequencies(qasm_file_path1)
        freq2 = get_gate_frequencies(qasm_file_path2)
        
        all_gates = set(freq1.keys()).union(set(freq2.keys()))
        # Check if the gate types are the same
        if set(freq1.keys()) != set(freq2.keys()):
            return 0.0  # Directly return 0 if the gate types are different
        vec1 = [freq1.get(gate, 0) for gate in all_gates]
        vec2 = [freq2.get(gate, 0) for gate in all_gates]
        
        dot_product = sum([vec1[i] * vec2[i] for i in range(len(all_gates))])
        norm1 = sum([x ** 2 for x in vec1]) ** 0.5
        norm2 = sum([x ** 2 for x in vec2]) ** 0.5
        
        return dot_product / (norm1 * norm2)

    def compare_qasm_lcs(self,qasm_lines, target_qasm_lines):
        def lcs_length(X, Y):
            m = len(X)
            n = len(Y)
            L = [[0] * (n + 1) for i in range(m + 1)]

            for i in range(m + 1):
                for j in range(n + 1):
                    if i == 0 or j == 0:
                        L[i][j] = 0
                    elif X[i - 1].strip() == Y[j - 1].strip():
                        L[i][j] = L[i - 1][j - 1] + 1
                    else:
                        L[i][j] = max(L[i - 1][j], L[i][j - 1])

            return L[m][n]
        # Calculate the length of the longest common subsequence
        lcs_len = lcs_length(qasm_lines, target_qasm_lines)
        
        # Calculate similarity score based on the length of LCS over the total lines in target_qasm
        score = lcs_len / len(target_qasm_lines) if target_qasm_lines else 0
        return score

    def compare_qasm(self, qasm, target_qasm):
        def is_file_empty(file_path):
            return os.path.getsize(file_path) == 0

        if is_file_empty(qasm) or is_file_empty(target_qasm):
            return 0
        try:
            if self.compare_method == 'fidelity':
                # Calculate unitary matrices for both QASM files
                unitary1 = self.qasm_to_unitary(qasm)
                unitary2 = self.qasm_to_unitary(target_qasm)
                
                # Calculate fidelity
                fidelity_score = process_fidelity(unitary1,unitary2)
                return fidelity_score

            elif self.compare_method == 'seq_similarity':
                # Gate sequence similarity
                seq1 = self.qasm_to_gate_sequence(qasm)
                seq2 = self.qasm_to_gate_sequence(target_qasm)
                seq_similarity = self.gate_sequence_similarity(seq1, seq2)
                return seq_similarity

            elif self.compare_method == 'freq_similarity':
                # Gate frequency similarity
                freq_similarity = self.gate_frequency_similarity(qasm, target_qasm)
                return freq_similarity


            elif self.compare_method == 'l_by_l':
                with open(qasm, 'r') as file1, open(target_qasm, 'r') as file2:
                    qasm_lines = file1.readlines()
                    target_qasm_lines = file2.readlines()
                    
                    score = self.compare_qasm_lcs(qasm_lines,target_qasm_lines)
                
                return score

            elif self.compare_method == 'combined':
                # Gate sequence similarity
                seq1 = self.qasm_to_gate_sequence(qasm)
                seq2 = self.qasm_to_gate_sequence(target_qasm)
                seq_similarity = self.gate_sequence_similarity(seq1, seq2)
                
                # Gate frequency similarity
                freq_similarity = self.gate_frequency_similarity(qasm, target_qasm)
                
                with open(qasm, 'r') as file1, open(target_qasm, 'r') as file2:
                    qasm_lines = file1.readlines()
                    target_qasm_lines = file2.readlines()
                    
                    lcs_similarity = self.compare_qasm_lcs(qasm_lines,target_qasm_lines)
                
                # combined_score = (seq_similarity + freq_similarity + inter_section_score) / 3
                combined_score = (seq_similarity * freq_similarity * lcs_similarity)**(1/3)
                return combined_score


        except FileNotFoundError:
            print(f"Error: One of the files not found ({qasm} or {target_qasm}).")
            return 0
        except Exception as e:
            print(f"Error comparing QASM files: {str(e)}")
            return 0   

    def evaluate(self, individual, individual_index):
        qasm_dir = os.path.join(self.qasm_path, f"{self.algorithm_name}_{individual_index}")
        target_qasm_dir = "Circuits"
        
        # Calculate score for each QASM file
        scores = []
        for i in range(2, self.qubit_limit+1):
            qasm_file = os.path.join(qasm_dir, f"{self.algorithm_name}_{individual_index}_{i}.qasm")
            target_qasm_file = os.path.join(target_qasm_dir, f"{self.algorithm_name}_{i}.qasm")
             ## debug
            # print(qasm_file,target_qasm_file)
            score = self.compare_qasm(qasm_file, target_qasm_file)
            scores.append(score)
        
        # Return the average score
        
        average_score = sum(scores) / len(scores) if scores else 0
        return average_score
            

    def run(self):
        if not self.perform_crossover and not self.perform_mutation:
            print("Warning: Both crossover and mutation are disabled; the population will not evolve.")

        best_score = float('-inf')
        best_individual = None
        best_generation_index = -1

        # Initialize lists to store scores
        best_scores = []
        all_scores = []
        best_code=[]

        # Generate initial population once at the beginning
        population = self.generate_initial_population(self.pop_size)

        for generation in range(self.generations):
            start_time = time.time()

            # Save the current population state and QASM data
            self.save(population)
            self.save_qasm()

            # Evaluate fitness for each individual
            fitness_scores = [self.evaluate(individual, index) for index, individual in enumerate(population)]

            # Save all fitness scores for this generation
            all_scores.append(fitness_scores)

            # Sort population by fitness (descending order)
            sorted_population = sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)
            sorted_scores, next_generation = zip(*sorted_population)
            next_generation = list(next_generation)
            sorted_scores = list(sorted_scores)

            # Select the best individual and corresponding score
            best_individual = next_generation[0]
            best_score = sorted_scores[0]
            
            # Save the best score for this generation
            best_scores.append(best_score)

            # Find the index of the best individual in the original population
            best_individual_index = fitness_scores.index(best_score)
            
            # Print debugging information
            # print(f"Algorithm : {self.algorithm_name}  Generation {generation + 1}: Best score = {best_score}")

            # # If the best score is 1, stop the iteration
            # if best_score == 1:
            #     break
            new_population = []

            # Number of individuals to be generated by each method
            crossover_count = int(self.pop_size * self.crossover_rate) if self.perform_crossover == True else 0
            mutation_count = int(self.pop_size * self.mutation_rate) if self.perform_mutation == True else 0
            new_gen_count = int(self.pop_size * self.new_gen_rate)
            elite_count = self.pop_size - crossover_count - mutation_count - new_gen_count

            # Preserve elite individuals
            new_population.extend(next_generation[:elite_count])

            # Apply crossover to generate new individuals
            while len(new_population) < elite_count + crossover_count:
                parent1, parent2 = self.select_parents(next_generation, sorted_scores, self.selection_method)
                
                child1, child2 = self.crossover(parent1, parent2)
            
                new_population.extend([child1, child2])

            # Apply mutation to new individuals
            for _ in range(mutation_count):
                if new_population:
                    individual_to_mutate = random.choice(new_population)
                    new_population.append(self.mutate(individual_to_mutate))
            
            # Generate new individuals
            new_population.extend(self.generate_initial_population(new_gen_count))
            individual_code = ast.unparse(best_individual) if best_individual else "No best individual found"
        
            best_code.append(individual_code)

            # Ensure the population size is correct after all operations
            # new_population = new_population[:self.pop_size]

            population = new_population

            # Apply annealing to the mutation_rate_2
            if self.perform_annealing:
                self.mutation_rate_2 = max(self.mutation_rate_2 * 0.99, 0.1)

            end_time = time.time()
            time_taken = end_time - start_time
            tqdm.write(f"Generation {generation + 1}/{self.generations} completed in {time_taken:.2f} seconds")

        # Unparse the AST of the best individual if found
        
        return best_code, best_scores, all_scores
   