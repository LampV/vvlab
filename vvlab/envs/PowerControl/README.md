# PowerControl  
This environment comes from the dissertation, 2020-Deep Reinforcement Learning for 5G Networks- Joint Beamforming, Power Control, and Interference Coordination.

On the basis of the original dissertation, code normalization is carried out, and code test and unit test are carried out respectively.

1. PC_env  

   In the case of two users and two base stations, the SNR of the user depends on the channel between the user and the base station, the transmitting power of the communication base station and the interference base station, and the beamforming mode.
   
   This environment standardizes the code in the original text, which creates an environment that can change the transmission power and beam forming mode of the base station.

2.  PC_env_format_test

      Test the formula in PC_env. Some formulas containing random numbers are not tested.

3.  PC_env_unit_test

      Unit test the code in PC_env.

4.  PC_env_origin

      The environment code in the original text.