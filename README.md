
# 🎨 Genetic Algorithm Art Generator

This repository contains a project that generates art using a genetic algorithm. The program evolves abstract representations of a target image using triangles and genetic operators like mutation and crossover.

---

## 📁 Project Contents

- **genetics.py** - The main Python file containing the genetic algorithm implementation for image generation.

---

## 🖌️ Project Features

- **Genetic Algorithm:**
  - **Chromosome Representation:** Each chromosome is a collection of triangles representing an image.
  - **Fitness Function:** Mean Squared Error (MSE) between the generated image and the target image.
  - **Selection Methods:** Adaptive Tournament Selection and Roulette Wheel Selection.
  - **Crossover Methods:**
    - Uniform Crossover
    - Single-Point Crossover
    - Two-Point Crossover
  - **Mutation Methods:**
    - Color, position, and size mutation for individual triangles.

- **Visualization:**
  - Displays the best chromosome's image during evolution.
  - Logs fitness statistics for each generation.

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/YourRepo.git
   ```

2. Navigate to the project directory:
   ```bash
   cd YourRepo
   ```

3. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the project:
   ```bash
   python3 genetics.py
   ```

---

## ⚙️ Requirements

- **Python 3.8+**
- **Libraries:**
  - numpy
  - matplotlib
  - opencv-python
  - Pillow

---

## 📊 How It Works

1. **Target Image:** The program loads a target image and resizes it.
2. **Initialization:** Random triangles are generated as the initial population.
3. **Evolution Process:**
   - **Selection:** Best chromosomes are selected based on fitness.
   - **Crossover:** Parent chromosomes produce offspring.
   - **Mutation:** Offspring undergo mutation for diversity.
   - **Evaluation:** The fitness of the new population is calculated.

4. **Output:**
   - The best image generated at various stages is displayed.
   - Final evolved image closely matching the target image.

