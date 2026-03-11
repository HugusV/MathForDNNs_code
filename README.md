# Presentation 

Repository containing all my experimental works on Deep Reinforcement Learning for path optimization for self-driven cars, in ESILV Research Track in Math for DNNs.  
My main work relies applying an ASAC algorithm (Adaptive Soft Actor Critic) which combines the DRL SAC (the core algorithm), DWA and Lyapunov stability approaches to solve complex navigation tasks. 
In some experimental code files, the Hybrid A* global planner is added to the ASAC algorithm to improve its navigation in statical or dynamical environments.
The robot kinematic model used for the DRL implementation is called Ackermann, the car-type model.  

See in priority the train and test codes of ASAC and ASAC + Hybrid A*.
