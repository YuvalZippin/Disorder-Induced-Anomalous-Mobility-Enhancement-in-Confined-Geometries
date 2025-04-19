# 🚀 Disorder-Induced Anomalous Mobility Enhancement in Confined Geometries

## 📄 Abstract
This project explores how spatial disorder can paradoxically enhance the mobility of particles within confined geometries. We reconstruct and extend Quantum Transport Models (QTMs) to examine the effects of quenched disorder and geometric constraints on particle dynamics. Our simulations aim to reproduce established results, verify analytical predictions, and contribute insights into anomalous diffusion in disordered environments.

## 🧪 Research Context
- **Laboratory**: Professor Stas Borv’s Research Group  
- **Researcher**: Undergraduate Physics Student (Static Physics specialization)  
- **Primary Focus**: Anomalous transport in disordered and confined media using Quantum Transport Models (QTMs)

## 🛠️ Methodology and Tools
- **Programming Languages**: Python (prototyping), C++ (high-performance implementation)
- **Libraries**: NumPy (numerics), Matplotlib (visualization), SciPy (scientific computing)

## 📁 Repository Structure
The code is modularly organized. Each script represents a distinct QTM variant (1D, 2D, 3D) or a utility model (WTD), allowing easy testing and result comparison.

---

## 🧭 `1D_QTM.py`: One-Dimensional Quenched Trap Model
Simulates a 1D QTM where particles experience site-dependent waiting times from a power-law distribution. The particle performs a biased random walk.

### 📐 Physical Model
- **Waiting Times**: Power-law distributed, fixed per site (quenched disorder)
- **Bias**: Configurable with `prob_right`
- **Start**: Centered at \( x = 0 \)

### ⏱️ Waiting Time Function
```python
func_transform(x) = x**(-2)
```
Induces subdiffusive behavior through long waiting times.

### 🔧 Functions
- `RW_sim_fixed_wait(...)`: Single random walk
- `multi_RW_sim_fixed_wait(...)`: Multiple trajectories
- `view_hist_fixed_wait(...)`: Histogram of final positions
- `second_moment_with_noise_fixed_wait(...)`: Noisy ⟨x²⟩
- `second_moment_without_noise_comp_fixed_wait(...)`: Clean ⟨x²⟩ with fit
- `second_moment_without_noise_comp_and_find_func(...)`: Extracts ⟨x²⟩ = A·tᵇ coefficients

### 📋 CLI Menu
```text
1. View Single Random Walk
2. View Histogram of Final Positions
3. View Second Moment with Noise
4. View Mean Second Moment with Power-Law Fit
5. Fit Power-Law Function f(t) = A·tᵇ
```

### 🧪 Example Parameters
```python
num_sims = 1000
sim_time = 1000
prob_right = 0.5
wait_list_size = 500
```

---

## 🌀 `2D_QTM.py`: Two-Dimensional Quenched Trap Model
Extends the QTM to 2D. The particle begins at the center and evolves over a 2D grid with periodic boundary conditions in the y-direction.

### 📐 Model Specifics
- **Waiting Time Field**: Static 2D grid generated from Lévy-stable distribution
- **Start**: Centered grid, (0,0) site has no waiting time
- **Motion**: Binomial random walk

### 🔧 Functions
- `run_single_trajectory(...)`: Single 2D particle path
- `plot_trajectory(...)`: Path visualization
- `final_position_distribution(...)`: End location histogram
- `compute_second_moment(...)`: Ensemble-averaged ⟨x²⟩
- `fit_power_law(...)`: Extracts ⟨x²⟩ = A·tᵇ for various widths W

### 🧪 Research Aims
- Analyze ⟨x²⟩ behavior across confinement widths W
- Understand subdiffusion persistence and enhancement

### 🌐 Observations
Preliminary results suggest disorder geometry crucially influences anomalous mobility trends.

---

## 🌐 `3D_QTM.py` *(Work in Progress)*: Three-Dimensional Quenched Trap Model
This evolving model expands the simulation to 3D, incorporating geometric constraints and directional movement probabilities.

### 🧱 Model Design
- **Waiting Time Distribution**: Heavy-tailed Lévy-type
- **Bias**: Set via `prob_vec` in [x, y, z] directions
- **Boundaries**: Periodic in Y and Z; open in X

### 🔧 Functions (in development)
- `run_single_trajectory(...)`: Particle path under full 3D bias
- `mean_final_position(...)`: Computes ⟨x⟩ over N trajectories
- `analyze_effect_of_alpha(...)`: Studies α-dependence of transport

### 🚧 In Progress
- Optimizing memory allocation and site reuse for faster simulations
- Comparing 2D and 3D confinement behavior

---

## ⏳ `WTD_model.py`: Waiting Time Distribution Analysis in 3D Confined Media
A general tool for analyzing confined QTM with custom waiting time fields. Designed for benchmarking and testing how confinement affects diffusion.

### ⏱️ Waiting Time Distribution
```python
S_alpha = lambda alpha: (1 / np.random.random())**(1 / alpha)
```
Implements a Lévy-type distribution:
\[ \psi(t) = \frac{\alpha}{t^{1+\alpha}}, \quad t \geq 1 \]

### 🔧 Functions
- `run_single_trajectory(...)`: Random walk under disorder
- `mean_final_position(...)`: Computes ensemble ⟨x⟩
- `plot_mean_position_vs_width(...)`: ⟨x⟩ as a function of width W
- `test_runtime_and_scaling(...)`: Performance benchmark

### 🎯 Research Goals
- Measure mobility enhancement under spatial constraints
- Test scalability across α and W ranges
- Lay groundwork for comparison with theoretical scaling laws

---

## 📌 How to Run
```bash
python 1D_QTM.py       # 1D Quenched Trap Model
python 2D_QTM.py       # 2D Confined Mobility Simulation
python 3D_QTM.py       # 3D Particle Transport (WIP)
python WTD_model.py    # Confined Lévy Transport Utility
```

---

## 📚 References and Theoretical Foundations
This project draws upon contemporary research in:
- Lévy flight models
- Quenched disorder in random media
- Anomalous diffusion under geometric constraints

Full citations and derivations are available in the extended report.

---

