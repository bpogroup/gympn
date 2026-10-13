# Documentation Overview

This document provides a quick index of all available documentation and what each covers.

## Documentation Structure

### Entry Points (Start Here!)

**Start here based on your learning style:**

1. **[Gentle Introduction](./gentle_introduction.md)** - New users
   - Basic concepts of Petri Nets and Action-Evolution Petri Nets
   - How to create your first environment
   - Step-by-step tutorial
   - Perfect for: Complete beginners

2. **[Complete Example](./complete_example.md)** - Practical learners
   - Full end-to-end example with all components
   - Real-world scenario (Supply Chain)
   - Shows best practices
   - Perfect for: Those who learn by doing

3. **[Algorithms Guide](./algorithms.md)** - Algorithm enthusiasts
   - Detailed explanation of PPO-Clip, PPO-Penalty and PG
   - Hyperparameter tuning guide
   - When to use each algorithm
   - Perfect for: Understanding algorithm details

### Core Concepts

These documents explain key concepts and features:

**[Agents Guide](./agents_guide.md)**
- PPOAgent: Standard policy gradient agent
- Custom solver creation
- Agent comparison and selection
- When to use which agent
- ~40 minutes read

**[Advanced Features](./advanced_features.md)**
- Graph Neural Networks (GNNs) for observations
- Parallelization and performance optimization
- Custom reward functions
- ~45 minutes read

### Specialized Topics

**[Net-Factored Credit (NF-GAE)](./nfgae.md)** ⭐ Main method
- Why whole-net rewards are noise for decisions in independent parts
- Net components and how gympn derives them from the net
- The per-component SMDP-GAE estimator and its conditions
- A worked two-component example
- ~30 minutes read

**[Postponement and Net-Factored Credit](./postponement.md)** ⭐ Most Advanced
- Postponement feature: defer decisions strategically
- Net-factored credit (NF-GAE) for nets with independent parts
- Implementation details
- Use cases and best practices
- ~60 minutes read

**[Extending GymPN](./extending_gympn.md)**
- Create custom environments
- Custom Petri Net models
- Custom policy networks
- Custom solvers
- Integration patterns
- ~50 minutes read

### Practical Guides

**[API Quick Reference](./api_reference.md)**
- Quick lookup for classes and functions
- Common parameters and options
- Return values and exceptions
- Code snippets for common tasks
- Perfect for: Quick lookups while coding

**[Benchmarks](./benchmarks.md)**
- Algorithm performance comparison
- Hyperparameter sensitivity
- Hardware requirements
- Scaling recommendations
- Use-case specific recommendations
- Perfect for: Choosing algorithms and settings

**[Troubleshooting](./troubleshooting.md)**
- Common errors and solutions
- Debugging techniques
- Performance tips
- TensorBoard/W&B setup
- FAQ
- Perfect for: Problem solving

### Contributing

**[Contributing Guide](./contributing.md)**
- How to contribute to GymPN
- Code standards
- Testing requirements
- Documentation guidelines
- PR process

---

## Reading Paths

### Path 1: Learning for the First Time ⭐ Recommended
1. [Gentle Introduction](./gentle_introduction.md) - 30 min
2. [Complete Example](./complete_example.md) - 40 min
3. [API Quick Reference](./api_reference.md) - 15 min (bookmark for later)
4. Start experimenting!
5. When you need advanced features, read [Advanced Features](./advanced_features.md)

**Total time: ~1.5 hours** to get started

### Path 2: Deep Learning (Understanding Everything)
1. [Gentle Introduction](./gentle_introduction.md) - 30 min
2. [Algorithms Guide](./algorithms.md) - 40 min
3. [Agents Guide](./agents_guide.md) - 40 min
4. [Advanced Features](./advanced_features.md) - 45 min
5. [Postponement and Net-Factored Credit](./postponement.md) - 60 min
6. [Extending GymPN](./extending_gympn.md) - 50 min

**Total time: ~5 hours** for comprehensive understanding

### Path 3: Quick Setup for Production
1. [Complete Example](./complete_example.md) - 40 min
2. [Benchmarks](./benchmarks.md) - 20 min (select hyperparameters)
3. [API Quick Reference](./api_reference.md) - 15 min
4. [Troubleshooting](./troubleshooting.md) - 10 min (bookmark for later)
5. Start using!

**Total time: ~1.5 hours** to production-ready

### Path 4: Advanced Features Only
- [Algorithms Guide](./algorithms.md) - for algorithm details
- [Agents Guide](./agents_guide.md) - for agent setup
- [Advanced Features](./advanced_features.md) - for GNNs and optimization
- [Postponement and Net-Factored Credit](./postponement.md) - for postponement and per-component credit
- [Extending GymPN](./extending_gympn.md) - for custom components

---

## Quick Reference by Use Case

### "I want to solve a task assignment problem"
→ [Complete Example](./complete_example.md) + [Benchmarks](./benchmarks.md)

### "I want to create a custom Petri Net model"
→ [Gentle Introduction](./gentle_introduction.md) + [Extending GymPN](./extending_gympn.md)

### "I want to understand the PPO variants"
→ [Algorithms Guide](./algorithms.md)

### "I want to optimize performance"
→ [Benchmarks](./benchmarks.md) + [Advanced Features](./advanced_features.md)

### "My training is unstable"
→ [Troubleshooting](./troubleshooting.md) + [Algorithms Guide](./algorithms.md)

### "My net has several independent processes"
→ [Postponement and Net-Factored Credit](./postponement.md) + [Agents Guide](./agents_guide.md)

### "I want to extend the framework"
→ [Extending GymPN](./extending_gympn.md) + [API Quick Reference](./api_reference.md)

### "I need quick answers"
→ [API Quick Reference](./api_reference.md) + [Troubleshooting](./troubleshooting.md)

---

## Document Details

| Document | Length | Difficulty | Topics |
|----------|--------|-----------|--------|
| Gentle Introduction | ~30 min | Beginner | Basics, first environment |
| Complete Example | ~40 min | Beginner | Full example, best practices |
| Algorithms | ~40 min | Intermediate | PPO variants, hyperparameters |
| Agents Guide | ~40 min | Intermediate | Agent selection, custom solvers |
| Advanced Features | ~45 min | Advanced | GNNs, parallelization, optimization |
| Postponement & NF-GAE | ~60 min | Advanced | Postponement, credit assignment |
| Extending GymPN | ~50 min | Advanced | Custom environments, networks |
| API Reference | ~15 min | Any | Quick lookups, code snippets |
| Benchmarks | ~20 min | Intermediate | Performance, recommendations |
| Troubleshooting | ~30 min | Any | Common issues, solutions |
| Contributing | ~20 min | Intermediate | How to contribute code |

---

## Key Concepts by Document

### Core Concepts
- **Action-Evolution Petri Nets (AEPN)**: Gentle Introduction, Complete Example
- **Gymnasium Integration**: Gentle Introduction, Extending GymPN
- **Graph Observations**: Advanced Features, API Reference

### Algorithms
- **Policy Gradient (PG)**: Algorithms, Benchmarks
- **Proximal Policy Optimization (PPO)**: Algorithms, Agents Guide, Benchmarks
- **PPO-Clip**: Algorithms, Agents Guide, Benchmarks
- **PPO-Penalty**: Algorithms, Agents Guide, Benchmarks

### Features
- **Postponement**: Postponement & NF-GAE, Extending GymPN
- **Net-factored credit**: Postponement & NF-GAE
- **Graph Neural Networks**: Advanced Features, Extending GymPN
- **Custom Solvers**: Agents Guide, Extending GymPN
- **Parallel Training**: Advanced Features, Benchmarks

### Practical
- **Environment Setup**: Gentle Introduction, Complete Example, Extending GymPN
- **Training**: Complete Example, Algorithms, Benchmarks
- **Hyperparameter Tuning**: Algorithms, Benchmarks
- **Debugging**: Troubleshooting, Advanced Features
- **Performance Optimization**: Advanced Features, Benchmarks

---

## Updated Frequently

These sections are updated regularly:
- **Benchmarks**: When new hardware configurations are tested
- **Troubleshooting**: When new issues are discovered and solved
- **API Reference**: When new classes/functions are added
- **Contributing**: When the contribution process changes

Last updated: January 2026
